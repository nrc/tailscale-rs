//! Generic implementations of the various storage operations.
//!
//! Operations are implemented on handle types which give access to exactly the data an operation
//! needs: a single table (`TableRef` and `TableMut`), a table accessed via one of its indexes
//! (`IndexRef` and `IndexMut`), or a single singleton (`SingletonRef` and `SingletonMut`). The
//! public accessor types (raw, owned, transactional, and read-only transactional) create these
//! handles and delegate to them.
//!
//! A mutable handle never covers more than the table or singleton it operates on (in particular,
//! never the whole store). That is what allows a transaction to give out mutable access to
//! different tables at the same time (see [`crate::TableTransaction`]).
//!
//! All mutating operations happen within a transaction; raw (non-transactional) mutations are
//! single-operation transactions.

use std::{borrow::Borrow, collections::HashMap, hash::Hash};

use crate::{
    Error, IndexIterator, Owner, Result, TableIterator,
    schema::{IndexDesc, SingletonDesc, TableDesc},
    storage::{self, Storage, Table, VersionedValue},
    transactions::TxnId,
};

pub(crate) type Base<T> = <T as IndexDesc>::BaseTable;
pub(crate) type BaseKey<T> = <<T as IndexDesc>::BaseTable as TableDesc>::Key;
pub(crate) type BaseValue<T> = <<T as IndexDesc>::BaseTable as TableDesc>::Value;
pub(crate) type IndexKey<T> = <T as TableDesc>::Key;
pub(crate) type IndexValue<T> = <T as TableDesc>::Value;

/// A base table (including its indexes) of the index `T`.
type BaseTable<T> = Table<Base<T>, <Base<T> as TableDesc>::IndexStorage>;

/// Read access to a single table, as seen by a transaction (or the latest committed state).
pub(crate) struct TableRef<'a, D: TableDesc> {
    table: &'a Table<D, D::IndexStorage>,
    txn_id: TxnId,
}

impl<D: TableDesc> Clone for TableRef<'_, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: TableDesc> Copy for TableRef<'_, D> {}

impl<'a, D: TableDesc> TableRef<'a, D> {
    pub(crate) fn new(table: &'a Table<D, D::IndexStorage>, txn_id: TxnId) -> Self {
        TableRef { table, txn_id }
    }

    /// Access `D`'s table in `storage` (which must not have a transaction in progress, or be
    /// accessed from that transaction).
    pub(crate) fn from_storage(storage: &'a Storage<D::Storage>) -> Self {
        TableRef::new(D::get_table(&storage.tables), storage.txn_id())
    }

    pub(crate) fn len(self) -> usize {
        self.table.len(self.txn_id)
    }

    pub(crate) fn is_empty(self) -> bool {
        self.table.is_empty(self.txn_id)
    }

    pub(crate) fn get<Q>(self, key: &Q) -> Option<&'a D::Value>
    where
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.table.get(key, self.txn_id)
    }

    pub(crate) fn iter<Kind>(self) -> TableIterator<'a, D, Kind> {
        TableIterator::new(self.table.iter(self.txn_id))
    }
}

/// Mutable access to a single table, within a transaction.
pub(crate) struct TableMut<'a, D: TableDesc> {
    table: &'a mut Table<D, D::IndexStorage>,
    txn_id: TxnId,
    max_committed_id: TxnId,
}

impl<'a, D: TableDesc> TableMut<'a, D> {
    pub(crate) fn new(
        table: &'a mut Table<D, D::IndexStorage>,
        txn_id: TxnId,
        max_committed_id: TxnId,
    ) -> Self {
        TableMut {
            table,
            txn_id,
            max_committed_id,
        }
    }

    pub(crate) fn clear(self, owner: Owner) {
        self.table.assert_owner(owner);
        self.table.clear(self.txn_id, self.max_committed_id);
    }

    pub(crate) fn insert(self, key: D::Key, value: D::Value, owner: Owner) {
        self.table.assert_owner(owner);
        self.table
            .insert(key, value, self.txn_id, self.max_committed_id);
    }

    pub(crate) fn with_mut<Q, T>(
        self,
        key: &Q,
        f: impl FnOnce(&mut D::Value) -> T,
        owner: Owner,
    ) -> Option<T>
    where
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + ToOwned<Owned = D::Key>,
        D::Value: Clone + PartialEq,
    {
        self.table.assert_owner(owner);
        self.table
            .with_mut(key, f, self.txn_id, self.max_committed_id)
    }

    pub(crate) fn remove<Q>(self, key: &Q, owner: Owner)
    where
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + ToOwned<Owned = D::Key>,
    {
        self.table.assert_owner(owner);
        self.table.remove(key, self.txn_id, self.max_committed_id);
    }

    /// Pass an iterator over the table giving mutable access to its values to `f`.
    ///
    /// Access is scoped by a closure (rather than returning an iterator) because the table's
    /// indexes must be rebuilt from the mutated values, which is only possible once no references
    /// to those values remain.
    pub(crate) fn with_iter_mut<F, T>(self, owner: Owner, f: F) -> T
    where
        F: for<'b> FnOnce(&mut dyn Iterator<Item = (&'b D::Key, &'b mut D::Value)>) -> T,
        D::Value: Clone + PartialEq,
    {
        self.table.assert_owner(owner);

        let mut yielded = Vec::new();
        let result = {
            let mut iter = self
                .table
                .iter_mut(self.txn_id, self.max_committed_id, |_| true)
                .inspect(|(k, _)| yielded.push((*k).clone()));
            f(&mut iter)
        };

        // The iterator has de-indexed the yielded rows, re-index them from their new values.
        for k in &yielded {
            self.table
                .rebuild_indexes_for_key(k, self.txn_id, self.max_committed_id);
        }

        result
    }
}

/// Read access to a table via the index `I`, as seen by a transaction (or the latest committed
/// state).
pub(crate) struct IndexRef<'a, I: IndexDesc> {
    base: &'a BaseTable<I>,
    txn_id: TxnId,
}

impl<I: IndexDesc> Clone for IndexRef<'_, I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<I: IndexDesc> Copy for IndexRef<'_, I> {}

impl<'a, I: IndexDesc> IndexRef<'a, I> {
    pub(crate) fn new(base: TableRef<'a, I::BaseTable>) -> Self {
        IndexRef {
            base: base.table,
            txn_id: base.txn_id,
        }
    }

    pub(crate) fn check_consistent(self) -> Result<()> {
        if I::index(self.base).is_poisoned(self.txn_id) {
            Err(Error::NonUniqueIndexKey(I::NAME))
        } else {
            Ok(())
        }
    }

    pub(crate) fn get<Q>(self, key: &Q) -> Result<(&'a BaseKey<I>, &'a BaseValue<I>)>
    where
        IndexKey<I>: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.check_consistent()?;
        let base_key = I::index(self.base)
            .get(key, self.txn_id)
            .ok_or(Error::NotPresent)?;
        let value = self
            .base
            .get(base_key, self.txn_id)
            .ok_or(Error::NotPresent)?;
        Ok((base_key, value))
    }

    pub(crate) fn iter<Kind>(self) -> IndexIterator<'a, I, Kind> {
        IndexIterator::new(
            self.base,
            I::index(self.base).iter(self.txn_id),
            self.txn_id,
        )
    }
}

/// Mutable access to a table via the index `I`, within a transaction.
pub(crate) struct IndexMut<'a, I: IndexDesc> {
    base: &'a mut BaseTable<I>,
    txn_id: TxnId,
    max_committed_id: TxnId,
}

impl<'a, I: IndexDesc> IndexMut<'a, I> {
    pub(crate) fn new(base: TableMut<'a, I::BaseTable>) -> Self {
        IndexMut {
            base: base.table,
            txn_id: base.txn_id,
            max_committed_id: base.max_committed_id,
        }
    }

    pub(crate) fn with_mut<Q, T>(
        self,
        key: &Q,
        f: impl FnOnce(&BaseKey<I>, &mut BaseValue<I>) -> T,
        owner: Owner,
    ) -> Result<T>
    where
        IndexKey<I>: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
        BaseValue<I>: Clone + PartialEq,
    {
        if I::index(self.base).is_poisoned(self.txn_id) {
            return Err(Error::NonUniqueIndexKey(I::NAME));
        }
        self.base.assert_owner(owner);

        // Cloned because the key is borrowed from the index, which is part of the base table and
        // is updated by `with_mut`.
        let base_key = I::index(self.base)
            .get(key, self.txn_id)
            .ok_or(Error::NotPresent)?
            .clone();
        self.base
            .with_mut(
                &base_key,
                |v| f(&base_key, v),
                self.txn_id,
                self.max_committed_id,
            )
            .ok_or(Error::NotPresent)
    }

    pub(crate) fn remove<Q>(self, key: &Q, owner: Owner)
    where
        IndexKey<I>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + ToOwned<Owned = IndexKey<I>>,
    {
        self.base.assert_owner(owner);

        // Cloned for the same reason as in `with_mut`.
        let Some(base_key) = I::index(self.base).get(key, self.txn_id).cloned() else {
            return;
        };
        self.base
            .remove(&base_key, self.txn_id, self.max_committed_id);
        I::index_mut(self.base).remove(key, self.txn_id, self.max_committed_id);
    }

    /// Pass an iterator over the index giving mutable access to the base table's values to `f`.
    ///
    /// Each row of the base table which has at least one key in the index is yielded once (with
    /// one of its index keys). See `TableMut::with_iter_mut` for why access is scoped by a closure.
    pub(crate) fn with_iter_mut<F, T>(self, owner: Owner, f: F) -> T
    where
        F: for<'b> FnOnce(
            &mut dyn Iterator<Item = (&'b IndexKey<I>, &'b BaseKey<I>, &'b mut BaseValue<I>)>,
        ) -> T,
        BaseValue<I>: Clone + PartialEq,
    {
        self.base.assert_owner(owner);

        // Iterate the base table rather than the index, since iterating the base table mutably
        // updates the index (to de-index the yielded rows). So we snapshot the index first.
        let mut index_keys = HashMap::new();
        for (index_key, base_key) in I::index(self.base).iter(self.txn_id) {
            index_keys
                .entry(base_key.clone())
                .or_insert_with(|| index_key.clone());
        }

        let mut yielded = Vec::new();
        let result = {
            let index_keys = &index_keys;
            let yielded = &mut yielded;
            let mut iter = self
                .base
                .iter_mut(self.txn_id, self.max_committed_id, |base_key| {
                    index_keys.contains_key(base_key)
                })
                .map(move |(base_key, value)| {
                    yielded.push(base_key.clone());
                    (&index_keys[base_key], base_key, value)
                });
            f(&mut iter)
        };

        // The iterator has de-indexed the yielded rows, re-index them from their new values.
        for k in &yielded {
            self.base
                .rebuild_indexes_for_key(k, self.txn_id, self.max_committed_id);
        }

        result
    }
}

/// Read access to a singleton, as seen by a transaction (or the latest committed state).
pub(crate) struct SingletonRef<'a, S: SingletonDesc> {
    value: &'a VersionedValue<Option<S::Value>>,
    txn_id: TxnId,
}

impl<'a, S: SingletonDesc> SingletonRef<'a, S> {
    pub(crate) fn new(value: &'a VersionedValue<Option<S::Value>>, txn_id: TxnId) -> Self {
        SingletonRef { value, txn_id }
    }

    /// Access `S` in `storage` (which must not have a transaction in progress, or be accessed from
    /// that transaction).
    pub(crate) fn from_storage(storage: &'a Storage<S::Storage>) -> Self {
        SingletonRef::new(S::get_ref(&storage.tables), storage.txn_id())
    }

    pub(crate) fn get(self) -> Option<&'a S::Value> {
        self.value.get(self.txn_id)?.as_ref()
    }
}

/// Mutable access to a singleton, within a transaction.
pub(crate) struct SingletonMut<'a, S: SingletonDesc> {
    value: &'a mut VersionedValue<Option<S::Value>>,
    txn_id: TxnId,
}

impl<'a, S: SingletonDesc> SingletonMut<'a, S> {
    pub(crate) fn new(value: &'a mut VersionedValue<Option<S::Value>>, txn_id: TxnId) -> Self {
        SingletonMut { value, txn_id }
    }

    pub(crate) fn insert(self, value: S::Value, owner: Owner) {
        assert_owner::<S>(owner);
        self.value.set(Some(value), self.txn_id);
    }

    pub(crate) fn remove(self, owner: Owner) {
        assert_owner::<S>(owner);
        self.value.set(None, self.txn_id);
    }

    pub(crate) fn with_mut<T>(self, f: impl FnOnce(&mut S::Value) -> T, owner: Owner) -> Option<T>
    where
        S::Value: Clone + PartialEq,
    {
        assert_owner::<S>(owner);
        storage::with_mut_singleton(self.value, self.txn_id, f)
    }
}

#[allow(unused_variables)]
#[track_caller]
fn assert_owner<D: SingletonDesc>(owner: Owner) {
    #[cfg(debug_assertions)]
    assert_eq!(
        D::OWNER,
        owner,
        "Ownership violation: expected {}, found {owner}",
        D::OWNER,
    );
}
