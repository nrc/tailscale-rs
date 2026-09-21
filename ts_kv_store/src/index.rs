use std::{borrow::Borrow, hash::Hash, marker::PhantomData};

use crate::{
    KvStore, Owner, Result, RoTableTransaction, TableTransaction,
    iter::{Keys, KeysAndValues},
    operations::{BaseKey, BaseValue, IndexMut, IndexRef, IndexValue, TableRef},
    schema::{IndexDesc, TableDesc},
    transactions::SchemaTransaction,
};

/// Apply `f` to `D`'s view of a transaction which contains only that operation, and commit it.
///
/// Returns the error from committing, if there is one. This is how the non-transactional index
/// operations get their atomicity: each one is a single-operation transaction.
fn in_index_txn<D, T>(
    store: &KvStore<D::Storage>,
    owner: Owner,
    f: impl FnOnce(&mut IndexTransaction<'_, '_, D>) -> T,
) -> Result<T>
where
    D: IndexDesc,
{
    let mut txn = store.begin_transaction(owner);
    // An index has no field of its own in the generated transaction, so we reach the table via the
    // base table's view.
    let result = f(&mut IndexTransaction::new(
        <D::BaseTable as TableDesc>::make_txn_view(&mut txn),
    ));
    txn.commit()?;
    Ok(result)
}

/// Apply `f` to the index `D` of `store` (not as part of any transaction).
fn read_index<D: IndexDesc, T>(
    store: &KvStore<D::Storage>,
    f: impl FnOnce(IndexRef<'_, D>) -> T,
) -> T {
    let guard = store.get_read_lock();
    f(IndexRef::new(TableRef::from_storage(&guard)))
}

/// An abstraction for operating on a table of key/values pairs via an index.
///
/// `Index` has no transactional semantics and only exists as a convenience for accessing tabular
/// data; each mutating operation is its own single-operation transaction.
///
/// `D` describes the index table, its base table is `D::BaseTable`.
///
/// There is a field of this type for each of a table's indexes in the struct returned by
/// [`Table::indexes`](crate::Table::indexes).
pub struct Index<'store, D: IndexDesc> {
    store: &'store KvStore<D::Storage>,
    desc: PhantomData<D>,
}

impl<'store, D: IndexDesc> Index<'store, D> {
    /// Create an index accessor for `store`.
    #[doc(hidden)]
    pub fn new(store: &'store KvStore<D::Storage>) -> Self {
        Index {
            store,
            desc: PhantomData,
        }
    }

    /// Returns `Ok` if the index is consistent, and an error with some kind of explanation if not.
    pub fn check_consistent(&self) -> Result<()> {
        read_index::<D, _>(self.store, |index| index.check_consistent())
    }

    /// Get a row of the table from the store by cloning the value.
    ///
    /// Returns `Error::NotPresent` if there is no value for the specified key.
    pub fn get<Q>(&self, _owner: Owner, key: &Q) -> Result<(BaseKey<D>, BaseValue<D>)>
    where
        BaseKey<D>: Clone,
        BaseValue<D>: Clone,
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
        IndexValue<D>: Eq + Hash,
    {
        read_index::<D, _>(self.store, |index| {
            index.get(key).map(|(k, v)| (k.clone(), v.clone()))
        })
    }

    /// Get immutable access to a row of the table in the store by reference.
    ///
    /// Returns `Error::NotPresent` (and does not call `f`) if there is no value for the specified key.
    pub fn with<Q, T>(
        &self,
        _owner: Owner,
        key: &Q,
        f: impl FnOnce(&BaseKey<D>, &BaseValue<D>) -> T,
    ) -> Result<T>
    where
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
        IndexValue<D>: Eq + Hash,
    {
        read_index::<D, _>(self.store, |index| index.get(key).map(|(k, v)| f(k, v)))
    }

    /// Get mutable access to a row of the table in the store in the store.
    ///
    /// Returns `Error::NotPresent` (and does not call `f`) if there is no value for the specified key.
    pub fn with_mut<Q, T>(
        &self,
        owner: Owner,
        key: &Q,
        f: impl FnOnce(&BaseKey<D>, &mut BaseValue<D>) -> T,
    ) -> Result<T>
    where
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
        BaseKey<D>: Clone,
        BaseValue<D>: Clone + PartialEq,
        IndexValue<D>: Eq + Hash,
    {
        in_index_txn::<D, _>(self.store, owner, |view| view.with_mut(key, f))?
    }

    /// Remove a row from the table.
    pub fn remove<Q>(&self, owner: Owner, key: &Q)
    where
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + ToOwned<Owned = D::Key>,
        IndexValue<D>: Eq + Hash + ToOwned<Owned = BaseKey<D>>,
    {
        // Should never panic since transaction should only fail on index inserts.
        in_index_txn::<D, _>(self.store, owner, |view| view.remove(key)).unwrap()
    }

    /// Pass an iterator over all the keys in the index and values in the base table to `f`.
    ///
    /// The store is locked for reading while `f` is called. Access is scoped by a closure (rather
    /// than returning an iterator) so that the yielded references cannot outlive the lock.
    pub fn with_iter<F, T>(&self, _owner: Owner, f: F) -> T
    where
        F: for<'a> FnOnce(
            &mut dyn Iterator<Item = (&'a D::Key, &'a BaseKey<D>, &'a BaseValue<D>)>,
        ) -> T,
        IndexValue<D>: Eq + Hash,
    {
        read_index::<D, _>(self.store, |index| f(&mut index.iter::<KeysAndValues>()))
    }

    /// Pass an iterator over all the keys in the index to `f`.
    ///
    /// See [`Self::with_iter`] for why access is scoped by a closure.
    pub fn with_keys<F, T>(&self, _owner: Owner, f: F) -> T
    where
        F: for<'a> FnOnce(&mut dyn Iterator<Item = &'a D::Key>) -> T,
    {
        read_index::<D, _>(self.store, |index| f(&mut index.iter::<Keys>()))
    }

    /// Pass an iterator over all the key/value pairs in the table, with mutable access to the
    /// values, to `f`.
    pub fn with_iter_mut<F, T>(&self, owner: Owner, f: F) -> T
    where
        F: for<'a> FnOnce(
            &mut dyn Iterator<Item = (&'a D::Key, &'a BaseKey<D>, &'a mut BaseValue<D>)>,
        ) -> T,
        IndexValue<D>: Eq + Hash + Clone,
        BaseValue<D>: Clone + PartialEq,
    {
        // Should never panic since transaction should only fail on index inserts.
        in_index_txn::<D, _>(self.store, owner, |view| view.with_iter_mut(f)).unwrap()
    }
}

/// An abstraction for operating on a table of key/values pairs via an index, with a fixed owner.
///
/// The owner-carrying counterpart of [`Index`]: the operations are the same, but the owner is
/// supplied once (by [`TableWithOwner::indexes`](crate::TableWithOwner::indexes)) rather than on
/// each call.
///
/// `D` describes the index table, its base table is `D::BaseTable`.
pub struct IndexWithOwner<'store, D: IndexDesc> {
    inner: Index<'store, D>,
    owner: Owner,
}

impl<'store, D: IndexDesc> IndexWithOwner<'store, D> {
    #[doc(hidden)]
    pub fn new(store: &'store KvStore<D::Storage>, owner: Owner) -> Self {
        IndexWithOwner {
            inner: Index::new(store),
            owner,
        }
    }

    /// Returns `Ok` if the index is consistent, and an error with some kind of explanation if not.
    pub fn check_consistent(&self) -> Result<()> {
        self.inner.check_consistent()
    }

    /// Get a row of the table from the store by cloning the value.
    ///
    /// Returns `Error::NotPresent` if there is no value for the specified key.
    pub fn get<Q>(&self, key: &Q) -> Result<(BaseKey<D>, BaseValue<D>)>
    where
        BaseKey<D>: Clone,
        BaseValue<D>: Clone,
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
        IndexValue<D>: Eq + Hash,
    {
        self.inner.get(self.owner, key)
    }

    /// Get immutable access to a row of the table in the store by reference.
    ///
    /// Returns `Error::NotPresent` (and does not call `f`) if there is no value for the specified key.
    pub fn with<Q, T>(&self, key: &Q, f: impl FnOnce(&BaseKey<D>, &BaseValue<D>) -> T) -> Result<T>
    where
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
        IndexValue<D>: Eq + Hash,
    {
        self.inner.with(self.owner, key, f)
    }

    /// Get mutable access to a row of the table in the store in the store.
    ///
    /// Returns `Error::NotPresent` (and does not call `f`) if there is no value for the specified key.
    pub fn with_mut<Q, T>(
        &self,
        key: &Q,
        f: impl FnOnce(&BaseKey<D>, &mut BaseValue<D>) -> T,
    ) -> Result<T>
    where
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
        BaseKey<D>: Clone,
        BaseValue<D>: Clone + PartialEq,
        IndexValue<D>: Eq + Hash,
    {
        self.inner.with_mut(self.owner, key, f)
    }

    /// Remove a row from the table.
    pub fn remove<Q>(&self, key: &Q)
    where
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + ToOwned<Owned = D::Key>,
        IndexValue<D>: Eq + Hash + ToOwned<Owned = BaseKey<D>>,
    {
        self.inner.remove(self.owner, key)
    }

    /// Pass an iterator over all the keys in the index and values in the base table to `f`.
    ///
    /// See [`Index::with_iter`] for why access is scoped by a closure.
    pub fn with_iter<F, T>(&self, f: F) -> T
    where
        F: for<'a> FnOnce(
            &mut dyn Iterator<Item = (&'a D::Key, &'a BaseKey<D>, &'a BaseValue<D>)>,
        ) -> T,
        IndexValue<D>: Eq + Hash,
    {
        self.inner.with_iter(self.owner, f)
    }

    /// Pass an iterator over all the keys in the index to `f`.
    ///
    /// See [`Index::with_iter`] for why access is scoped by a closure.
    pub fn with_keys<F, T>(&self, f: F) -> T
    where
        F: for<'a> FnOnce(&mut dyn Iterator<Item = &'a D::Key>) -> T,
    {
        self.inner.with_keys(self.owner, f)
    }

    /// Pass an iterator over all the key/value pairs in the table, with mutable access to the
    /// values, to `f`.
    pub fn with_iter_mut<F, T>(&self, f: F) -> T
    where
        F: for<'a> FnOnce(
            &mut dyn Iterator<Item = (&'a D::Key, &'a BaseKey<D>, &'a mut BaseValue<D>)>,
        ) -> T,
        IndexValue<D>: Eq + Hash + Clone,
        BaseValue<D>: Clone + PartialEq,
    {
        self.inner.with_iter_mut(self.owner, f)
    }
}

/// An abstraction for operating on a table of key/values pairs (accessed as part of a transaction)
/// via an index.
///
/// The transactional counterpart of [`Index`]: the operations are the same, but they are part of
/// the transaction this was created from rather than being atomic on their own, and the owner comes
/// from the transaction.
///
/// `D` describes the index table, its base table is `D::BaseTable`.
///
/// Created by the methods of the struct returned by
/// [`TableTransaction::indexes`](crate::TableTransaction::indexes). An `IndexTransaction` mutably
/// borrows the base table's view for `'txn`, so only one index of a table can be used at a time.
pub struct IndexTransaction<'guard, 'txn, D: IndexDesc> {
    base: &'txn mut TableTransaction<'guard, D::Storage, D::BaseTable>,
    desc: PhantomData<D>,
}

impl<'guard, 'txn, D: IndexDesc> IndexTransaction<'guard, 'txn, D> {
    /// Create a view of the index `D` of the table viewed by `base`.
    #[doc(hidden)]
    pub fn new(base: &'txn mut TableTransaction<'guard, D::Storage, D::BaseTable>) -> Self {
        IndexTransaction {
            base,
            desc: PhantomData,
        }
    }

    fn index_ref(&self) -> IndexRef<'_, D> {
        IndexRef::new(self.base.table_ref())
    }

    fn index_mut(&mut self) -> IndexMut<'_, D> {
        IndexMut::new(self.base.table_mut())
    }

    /// Returns `Ok` if the index is consistent, and an error with some kind of explanation if not.
    pub fn check_consistent(&self) -> Result<()> {
        self.index_ref().check_consistent()
    }

    /// Get a row of the table from the store by cloning the value.
    ///
    /// Returns `Error::NotPresent` if there is no value for the specified key.
    pub fn get<Q>(&self, key: &Q) -> Result<(BaseKey<D>, BaseValue<D>)>
    where
        BaseKey<D>: Clone,
        BaseValue<D>: Clone,
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
        IndexValue<D>: Eq + Hash,
    {
        self.index_ref()
            .get(key)
            .map(|(k, v)| (k.clone(), v.clone()))
    }

    /// Get immutable access to a row of the table in the store by reference.
    ///
    /// Returns `Error::NotPresent` (and does not call `f`) if there is no value for the specified key.
    pub fn with<Q, T>(&self, key: &Q, f: impl FnOnce(&BaseKey<D>, &BaseValue<D>) -> T) -> Result<T>
    where
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
        IndexValue<D>: Eq + Hash,
    {
        self.index_ref().get(key).map(|(k, v)| f(k, v))
    }

    /// Get mutable access to a row of the table in the store in the store.
    ///
    /// Returns `Error::NotPresent` (and does not call `f`) if there is no value for the specified key.
    pub fn with_mut<Q, T>(
        &mut self,
        key: &Q,
        f: impl FnOnce(&BaseKey<D>, &mut BaseValue<D>) -> T,
    ) -> Result<T>
    where
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
        BaseKey<D>: Clone,
        BaseValue<D>: Clone + PartialEq,
        IndexValue<D>: Eq + Hash,
    {
        let owner = self.base.txn_owner();
        self.index_mut().with_mut(key, f, owner)
    }

    /// Remove a row from the table.
    pub fn remove<Q>(&mut self, key: &Q)
    where
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + ToOwned<Owned = D::Key>,
        IndexValue<D>: Eq + Hash + ToOwned<Owned = BaseKey<D>>,
    {
        let owner = self.base.txn_owner();
        self.index_mut().remove(key, owner)
    }

    /// Iterate all the keys in the index and value in the base table.
    pub fn iter(&self) -> impl Iterator<Item = (&D::Key, &BaseKey<D>, &BaseValue<D>)>
    where
        IndexValue<D>: Eq + Hash,
    {
        self.index_ref().iter::<KeysAndValues>()
    }

    /// Iterate all the keys in the index.
    pub fn keys(&self) -> impl Iterator<Item = &D::Key> {
        self.index_ref().iter::<Keys>()
    }

    /// Iterate all the key/value pairs in a table, with mutable access to the values.
    ///
    /// Access is scoped by a closure rather than by returning an iterator, because the table's
    /// indexes are updated from the mutated values once `f` returns.
    pub fn with_iter_mut<F, T>(&mut self, f: F) -> T
    where
        F: for<'a> FnOnce(
            &mut dyn Iterator<Item = (&'a D::Key, &'a BaseKey<D>, &'a mut BaseValue<D>)>,
        ) -> T,
        IndexValue<D>: Eq + Hash + Clone,
        BaseValue<D>: Clone + PartialEq,
    {
        let owner = self.base.txn_owner();
        self.index_mut().with_iter_mut(owner, f)
    }
}

/// An abstraction for operating on a table of key/values pairs (accessed as part of a read-only
/// transaction) via an index.
///
/// The read-only counterpart of [`IndexTransaction`], with only the non-mutating operations.
///
/// `D` describes the index table, its base table is `D::BaseTable`.
///
/// There is a field of this type for each of a table's indexes in the struct returned by
/// [`RoTableTransaction::indexes`](crate::RoTableTransaction::indexes).
pub struct RoIndexTransaction<'guard, 'txn, D: IndexDesc> {
    base: &'txn RoTableTransaction<'guard, D::Storage, D::BaseTable>,
    desc: PhantomData<D>,
}

impl<'guard, 'txn, D: IndexDesc> RoIndexTransaction<'guard, 'txn, D> {
    /// Create a view of the index `D` of the table viewed by `base`.
    #[doc(hidden)]
    pub fn new(base: &'txn RoTableTransaction<'guard, D::Storage, D::BaseTable>) -> Self {
        RoIndexTransaction {
            base,
            desc: PhantomData,
        }
    }

    fn index_ref(&self) -> IndexRef<'txn, D> {
        IndexRef::new(self.base.table_ref())
    }

    /// Returns `Ok` if the index is consistent, and an error with some kind of explanation if not.
    pub fn check_consistent(&self) -> Result<()> {
        self.index_ref().check_consistent()
    }

    /// Get a row of the table from the store by cloning the value.
    ///
    /// Returns `Error::NotPresent` if there is no value for the specified key.
    pub fn get<Q>(&self, key: &Q) -> Result<(BaseKey<D>, BaseValue<D>)>
    where
        BaseKey<D>: Clone,
        BaseValue<D>: Clone,
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
        IndexValue<D>: Eq + Hash,
    {
        self.index_ref()
            .get(key)
            .map(|(k, v)| (k.clone(), v.clone()))
    }

    /// Get immutable access to a row of the table in the store by reference.
    ///
    /// Returns `Error::NotPresent` (and does not call `f`) if there is no value for the specified key.
    pub fn with<Q, T>(&self, key: &Q, f: impl FnOnce(&BaseKey<D>, &BaseValue<D>) -> T) -> Result<T>
    where
        D::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
        IndexValue<D>: Eq + Hash,
    {
        self.index_ref().get(key).map(|(k, v)| f(k, v))
    }

    /// Iterate all the keys in the index and value in the base table.
    pub fn iter(&self) -> impl Iterator<Item = (&D::Key, &BaseKey<D>, &BaseValue<D>)>
    where
        IndexValue<D>: Eq + Hash,
    {
        self.index_ref().iter::<KeysAndValues>()
    }

    /// Iterate all the keys in the index.
    pub fn keys(&self) -> impl Iterator<Item = &D::Key> {
        self.index_ref().iter::<Keys>()
    }
}

#[cfg(test)]
mod test {
    use crate::{KvErrorExt, store};

    #[derive(Clone, Debug, PartialEq)]
    pub struct Row {
        pub name: String,
    }

    fn row(name: &str) -> Row {
        Row {
            name: name.to_owned(),
        }
    }

    store!(tables: { Users(u32 => Row; OWNER; index(name: String)) });

    const OWNER: &str = "owner";
    const OTHER: &str = "other";

    #[test]
    fn index_get_returns_none_when_absent() {
        let store = KvStore::new();
        assert!(store.Users.indexes().name.get(OWNER, "Alice").is_none());
    }

    #[test]
    fn index_get_returns_value_after_base_insert() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        let table = store.with_owner(OWNER).Users.indexes().name;
        let value = table.get("Alice").unwrap();
        assert_eq!(value, (1, row("Alice")));
    }

    #[test]
    fn index_with_returns_none_and_does_not_call_f_when_absent() {
        let store = KvStore::new();
        let mut called = false;
        let result = store.Users.indexes().name.with(OWNER, "Alice", |_, _| {
            called = true;
        });
        assert!(result.is_none());
        assert!(!called);
    }

    #[test]
    fn index_with_returns_result_of_f() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        let len = store.Users.indexes().name.with(OWNER, "Alice", |k, v| {
            assert_eq!(*k, 1);
            v.name.len()
        });
        assert_eq!(len.unwrap_opt(), Some(5));
    }

    #[test]
    fn base_insert_is_visible_via_index() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        assert!(store.Users.indexes().name.get(OWNER, "Alice").is_some());
    }

    #[test]
    fn index_remove_makes_base_absent() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        store.Users.indexes().name.remove(OWNER, "Alice");
        assert!(store.Users.get(OWNER, &1).is_none());
    }

    #[test]
    fn index_remove_makes_index_absent() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        store.Users.indexes().name.remove(OWNER, "Alice");
        assert!(store.Users.indexes().name.get(OWNER, "Alice").is_none());
    }

    #[test]
    fn base_remove_makes_index_absent() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        store.Users.remove(OWNER, &1);
        assert!(store.Users.indexes().name.get(OWNER, "Alice").is_none());
    }

    #[test]
    fn index_mutate_returns_none_when_absent() {
        let store = KvStore::new();
        let result = store
            .Users
            .indexes()
            .name
            .with_mut(OWNER, "Alice", |_, v| v.name.len());
        assert!(result.is_none());
    }

    #[test]
    fn index_mutate_modifies_value() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        store
            .Users
            .indexes()
            .name
            .with_mut(OWNER, "Alice", |k, v| {
                assert_eq!(*k, 1);
                v.name.push_str(" Smith")
            })
            .unwrap();
        let value = store.Users.get(OWNER, &1).unwrap();
        assert_eq!(value.name, "Alice Smith");
    }

    #[test]
    fn index_mutate_updates_index_on_field_change() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        store
            .Users
            .indexes()
            .name
            .with_mut(OWNER, "Alice", |_, v| {
                v.name = "Charlie".to_owned();
            })
            .unwrap();
        assert!(store.Users.indexes().name.get(OWNER, "Alice").is_none());
        let value = store.Users.indexes().name.get(OWNER, "Charlie").unwrap();
        assert_eq!(value, (1, row("Charlie")));
    }

    #[test]
    fn base_clear_removes_index_entries() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        store.Users.clear(OWNER);
        assert!(store.Users.indexes().name.get(OWNER, "Alice").is_none());
    }

    #[test]
    fn index_iter_empty_on_fresh_store() {
        let store = KvStore::new();
        let index = store.with_owner(OWNER).Users.indexes().name;
        let items: Vec<_> = index.iter().collect();
        assert!(items.is_empty());
    }

    #[test]
    fn index_iter_yields_index_key_and_base_value() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        let items: Vec<_> = store
            .Users
            .indexes()
            .name
            .iter(OWNER)
            .map(|(k, bk, v)| (k.clone(), *bk, v.clone()))
            .collect();
        assert_eq!(items, vec![("Alice".to_owned(), 1, row("Alice"))]);
    }

    #[test]
    fn index_iter_yields_all_rows() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        store.Users.insert(OWNER, 2, row("Bob"));
        let mut items: Vec<_> = store
            .Users
            .indexes()
            .name
            .iter(OWNER)
            .map(|(k, bk, v)| (k.clone(), *bk, v.clone()))
            .collect();
        items.sort_by_key(|(k, ..)| k.clone());
        assert_eq!(
            items,
            vec![
                ("Alice".to_owned(), 1, row("Alice")),
                ("Bob".to_owned(), 2, row("Bob")),
            ]
        );
    }

    #[test]
    fn index_iter_keys_cloned_empty() {
        let store = KvStore::new();
        let table = store.with_owner(OWNER).Users.indexes().name;

        let keys: Vec<_> = table.keys().collect();
        assert!(keys.is_empty());
    }

    #[test]
    fn index_iter_keys_cloned_yields_index_keys() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        store.Users.insert(OWNER, 2, row("Bob"));

        let table = store.with_owner(OWNER).Users.indexes().name;
        let mut keys: Vec<_> = table.keys().collect();
        keys.sort();
        assert_eq!(keys, vec!["Alice", "Bob"]);
    }

    #[test]
    fn index_for_each_empty_calls_closure_zero_times() {
        let store = KvStore::new();
        let index = store.with_owner(OWNER).Users.indexes().name;
        let mut count = 0;
        index.iter().for_each(|_| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn index_for_each_yields_index_key_and_base_value() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        let index = store.with_owner(OWNER).Users.indexes().name;
        let mut items: Vec<_> = Vec::new();
        index
            .iter()
            .for_each(|(k, bk, v)| items.push((k.clone(), *bk, v.clone())));
        assert_eq!(items, vec![("Alice".to_owned(), 1, row("Alice"))]);
    }

    #[test]
    fn index_for_each_yields_all_rows() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        store.Users.insert(OWNER, 2, row("Bob"));
        let index = store.with_owner(OWNER).Users.indexes().name;
        let mut items: Vec<_> = Vec::new();
        index
            .iter()
            .for_each(|(k, bk, v)| items.push((k.clone(), *bk, v.clone())));
        items.sort_by_key(|(k, ..)| k.clone());
        assert_eq!(
            items,
            vec![
                ("Alice".to_owned(), 1, row("Alice")),
                ("Bob".to_owned(), 2, row("Bob")),
            ]
        );
    }

    #[test]
    fn index_with_iter_mut_modifies_base_values() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        let index = store.with_owner(OWNER).Users.indexes().name;
        index.with_iter_mut(|i| i.next().unwrap().2.name.push('!'));
        assert_eq!(store.Users.get(OWNER, &1), Some(row("Alice!")));
    }

    #[test]
    fn table_with_iter_mut_updates_index() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        store
            .Users
            .with_iter_mut(OWNER, |i| i.next().unwrap().1.name = "Charlie".to_owned());
        assert!(store.Users.indexes().name.get(OWNER, "Alice").is_none());
        assert_eq!(
            store.Users.indexes().name.get(OWNER, "Charlie").unwrap(),
            (1, row("Charlie")),
        );
    }

    #[test]
    fn index_with_iter_mut_updates_index() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        store
            .Users
            .indexes()
            .name
            .with_iter_mut(OWNER, |i| i.next().unwrap().2.name = "Charlie".to_owned());
        assert!(store.Users.indexes().name.get(OWNER, "Alice").is_none());
        assert_eq!(
            store.Users.indexes().name.get(OWNER, "Charlie").unwrap(),
            (1, row("Charlie"))
        );
    }

    #[test]
    fn index_with_iter_mut_empty_yields_none() {
        let store = KvStore::new();
        let index = store.with_owner(OWNER).Users.indexes().name;
        let count = index.with_iter_mut(|i| i.count());
        assert_eq!(count, 0);
    }

    // Mutating every row through a multi-row index iterator must rebuild the index for all of
    // them (the single-row tests can't catch aliasing or partial-rebuild bugs).
    #[test]
    fn index_with_iter_mut_updates_all_rows() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        store.Users.insert(OWNER, 2, row("Bob"));
        store.Users.indexes().name.with_iter_mut(OWNER, |i| {
            for (_, _, v) in i {
                v.name.push('!');
            }
        });
        let index = store.with_owner(OWNER).Users.indexes().name;
        assert!(index.get("Alice").is_none());
        assert!(index.get("Bob").is_none());
        assert_eq!(index.get("Alice!").unwrap(), (1, row("Alice!")));
        assert_eq!(index.get("Bob!").unwrap(), (2, row("Bob!")));
    }

    // Visiting a row without mutating it still tears down and rebuilds its index entry (via
    // `get_mut` -> `on_remove` then `rebuild_indexes_for_key` on drop), so a row left unchanged
    // must remain correctly indexed alongside one that was changed.
    #[test]
    fn index_with_iter_mut_visited_unmodified_row_stays_indexed() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        store.Users.insert(OWNER, 2, row("Bob"));
        store.Users.indexes().name.with_iter_mut(OWNER, |i| {
            for (_, base_key, v) in i {
                if *base_key == 1 {
                    v.name = "Zara".to_owned();
                }
            }
        });
        let index = store.with_owner(OWNER).Users.indexes().name;
        assert!(index.get("Alice").is_none());
        assert_eq!(index.get("Zara").unwrap(), (1, row("Zara")));
        // Bob was visited but not modified; its index entry must be intact.
        assert_eq!(index.get("Bob").unwrap(), (2, row("Bob")));
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "Ownership violation"))]
    fn index_remove_wrong_owner_panics() {
        let store = KvStore::new();
        store.Users.indexes().name.remove(OTHER, "Alice");
    }
}

#[cfg(test)]
mod test_two_indexes {
    use crate::{KvErrorExt, store};

    #[derive(Clone, Debug, PartialEq)]
    pub struct Person {
        pub email: String,
        pub username: Vec<u8>,
    }

    fn person(email: &str, username: &[u8]) -> Person {
        Person {
            email: email.to_owned(),
            username: username.to_owned(),
        }
    }

    store!(tables: { People(u32 => Person; OWNER; index(email: String); index(username: Vec<u8>)) });

    const OWNER: &str = "owner";

    #[test]
    fn both_indexes_queryable_after_base_insert() {
        let store = KvStore::new();
        store
            .People
            .insert(OWNER, 1, person("a@example.com", b"alice"));
        assert!(
            store
                .People
                .indexes()
                .email
                .get(OWNER, "a@example.com")
                .is_some()
        );
        assert!(
            store
                .People
                .indexes()
                .username
                .get(OWNER, b"alice".as_slice())
                .is_some()
        );
    }

    #[test]
    fn each_index_returns_correct_value() {
        let store = KvStore::new();
        store
            .People
            .insert(OWNER, 1, person("a@example.com", b"alice"));
        store
            .People
            .insert(OWNER, 2, person("b@example.com", b"bob"));

        assert_eq!(
            store
                .People
                .indexes()
                .email
                .get(OWNER, "a@example.com")
                .unwrap(),
            (1, person("a@example.com", b"alice"))
        );
        assert_eq!(
            store
                .People
                .indexes()
                .username
                .get(OWNER, b"bob".as_slice())
                .unwrap(),
            (2, person("b@example.com", b"bob"))
        );
    }

    #[test]
    fn base_remove_clears_both_indexes() {
        let store = KvStore::new();
        store
            .People
            .insert(OWNER, 1, person("a@example.com", b"alice"));
        store.People.remove(OWNER, &1);
        assert!(
            store
                .People
                .indexes()
                .email
                .get(OWNER, "a@example.com")
                .is_none()
        );
        assert!(
            store
                .People
                .indexes()
                .username
                .get(OWNER, b"alice".as_slice())
                .is_none()
        );
    }

    #[test]
    fn email_index_remove_clears_both_indexes() {
        let store = KvStore::new();
        store
            .People
            .insert(OWNER, 1, person("a@example.com", b"alice"));
        store.People.indexes().email.remove(OWNER, "a@example.com");
        assert!(
            store
                .People
                .indexes()
                .email
                .get(OWNER, "a@example.com")
                .is_none()
        );
        assert!(
            store
                .People
                .indexes()
                .username
                .get(OWNER, b"alice".as_slice())
                .is_none()
        );
    }

    #[test]
    fn username_index_remove_clears_both_indexes() {
        let store = KvStore::new();
        store
            .People
            .insert(OWNER, 1, person("a@example.com", b"alice"));
        store
            .People
            .indexes()
            .username
            .remove(OWNER, b"alice".as_slice());
        assert!(
            store
                .People
                .indexes()
                .email
                .get(OWNER, "a@example.com")
                .is_none()
        );
        assert!(
            store
                .People
                .indexes()
                .username
                .get(OWNER, b"alice".as_slice())
                .is_none()
        );
    }

    #[test]
    fn index_remove_removes_from_base_table() {
        let store = KvStore::new();
        store
            .People
            .insert(OWNER, 1, person("a@example.com", b"alice"));
        store.People.indexes().email.remove(OWNER, "a@example.com");
        assert!(store.People.get(OWNER, &1).is_none());
    }

    #[test]
    fn base_clear_clears_both_indexes() {
        let store = KvStore::new();
        store
            .People
            .insert(OWNER, 1, person("a@example.com", b"alice"));
        store.People.clear(OWNER);
        assert!(
            store
                .People
                .indexes()
                .email
                .get(OWNER, "a@example.com")
                .is_none()
        );
        assert!(
            store
                .People
                .indexes()
                .username
                .get(OWNER, b"alice".as_slice())
                .is_none()
        );
    }

    #[test]
    fn table_with_iter_mut_updates_both_indexes() {
        let store = KvStore::new();
        store
            .People
            .insert(OWNER, 1, person("a@example.com", b"alice"));
        store.People.with_iter_mut(OWNER, |i| {
            let v = &mut i.next().unwrap().1;
            v.email = "b@example.com".to_owned();
            v.username = b"bob".to_vec();
        });
        assert!(
            store
                .People
                .indexes()
                .email
                .get(OWNER, "a@example.com")
                .is_none()
        );
        assert!(
            store
                .People
                .indexes()
                .username
                .get(OWNER, b"alice".as_slice())
                .is_none()
        );
        assert!(
            store
                .People
                .indexes()
                .email
                .get(OWNER, "b@example.com")
                .is_some()
        );
        assert!(
            store
                .People
                .indexes()
                .username
                .get(OWNER, b"bob".as_slice())
                .is_some()
        );
    }

    #[test]
    fn email_index_with_iter_mut_updates_both_indexes() {
        let store = KvStore::new();
        store
            .People
            .insert(OWNER, 1, person("a@example.com", b"alice"));
        store.People.indexes().email.with_iter_mut(OWNER, |iter| {
            iter.for_each(|(_, _, v)| {
                v.email = "b@example.com".to_owned();
                v.username = b"bob".to_vec();
            });
        });
        assert!(
            store
                .People
                .indexes()
                .email
                .get(OWNER, "a@example.com")
                .is_none()
        );
        assert!(
            store
                .People
                .indexes()
                .username
                .get(OWNER, b"alice".as_slice())
                .is_none()
        );
        assert!(
            store
                .People
                .indexes()
                .email
                .get(OWNER, "b@example.com")
                .is_some()
        );
        assert!(
            store
                .People
                .indexes()
                .username
                .get(OWNER, b"bob".as_slice())
                .is_some()
        );
    }

    // The tests below exist to pin down which accessors may be held at the same time. They are as
    // much compile-time tests as run-time ones: if the fields of the generated `Indexes` struct
    // stop being disjoint borrows, these stop compiling.

    #[test]
    fn both_indexes_of_one_table_are_usable_at_once() {
        let store = KvStore::new();
        store
            .People
            .insert(OWNER, 1, person("a@example.com", b"alice"));

        let indexes = store.People.indexes();
        let email = indexes.email;
        let username = indexes.username;

        // A mutation made through one index is visible through the other while both are held.
        email
            .with_mut(OWNER, "a@example.com", |_, v| {
                v.username = b"bob".to_vec();
            })
            .unwrap();
        assert!(username.get(OWNER, b"alice".as_slice()).is_none());
        assert_eq!(
            username.get(OWNER, b"bob".as_slice()).unwrap(),
            (1, person("a@example.com", b"bob"))
        );
    }

    #[test]
    fn both_indexes_of_one_table_are_mutable_at_once_in_a_txn() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.People.insert(1, person("a@example.com", b"alice"));

        let mut indexes = txn.People.indexes();
        let email = &mut indexes.email;
        let username = &mut indexes.username;

        // A mutation made through one index is visible through the other while both are held.
        email
            .with_mut("a@example.com", |_, v| {
                v.username = b"bob".to_vec();
            })
            .unwrap();
        assert!(username.get(b"alice".as_slice()).is_none());
        assert_eq!(
            username.get(b"bob".as_slice()).unwrap(),
            (1, person("a@example.com", b"bob"))
        );

        txn.commit().unwrap();
        assert!(
            store
                .People
                .indexes()
                .username
                .get(OWNER, b"bob".as_slice())
                .is_some()
        );
    }
}

#[cfg(test)]
mod test_transactional_index {
    use crate::{KvErrorExt, store};

    #[derive(Clone, Debug, PartialEq)]
    pub struct Row {
        pub name: String,
        pub age: u32,
    }

    fn row(name: &str) -> Row {
        Row {
            name: name.to_owned(),
            age: 0,
        }
    }

    store!(tables: { Users(u32 => Row; OWNER; index(name: String)) });

    const OWNER: &str = "owner";
    const OTHER: &str = "other";

    #[test]
    fn txn_index_get_returns_none_when_absent() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        assert!(txn.Users.indexes().name.get("Alice").is_none());
    }

    #[test]
    fn txn_index_remove_is_visible_after_commit() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        assert_eq!(
            txn.Users.indexes().name.get("Alice").unwrap(),
            (1, row("Alice"))
        );
        txn.commit().unwrap();
        assert_eq!(
            store.Users.indexes().name.get(OWNER, "Alice").unwrap(),
            (1, row("Alice"))
        );

        let mut txn = store.begin_transaction(OWNER);
        txn.Users.indexes().name.remove("Alice");
        txn.commit().unwrap();
        assert!(store.Users.indexes().name.get(OWNER, "Alice").is_none());
    }

    #[test]
    fn txn_index_with_returns_some() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        assert_eq!(
            txn.Users
                .indexes()
                .name
                .with("Alice", |k, v| {
                    assert_eq!(*k, 1);
                    v.name.len()
                })
                .unwrap(),
            5
        );
    }

    #[test]
    fn txn_index_mutate_updates_index_on_field_change() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        txn.Users
            .indexes()
            .name
            .with_mut("Alice", |_, v| v.name = "Bob".to_owned())
            .unwrap();
        assert!(txn.Users.indexes().name.get("Alice").is_none());
        assert_eq!(
            txn.Users.indexes().name.get("Bob").unwrap(),
            (
                1,
                Row {
                    name: "Bob".to_owned(),
                    age: 0
                }
            )
        );
    }

    #[test]
    fn txn_index_mutate_non_indexed_field_preserves_index() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        txn.Users
            .indexes()
            .name
            .with_mut("Alice", |_, v| v.age = 42)
            .unwrap();
        assert_eq!(
            txn.Users.indexes().name.get("Alice").unwrap(),
            (
                1,
                Row {
                    name: "Alice".to_owned(),
                    age: 42
                }
            )
        );
    }

    #[test]
    fn txn_index_remove_removes_from_index() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        txn.Users.indexes().name.remove("Alice");
        assert!(txn.Users.indexes().name.get("Alice").is_none());
    }

    #[test]
    fn txn_index_remove_removes_from_base() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        txn.Users.indexes().name.remove("Alice");
        assert!(txn.Users.get(&1).is_none());
        txn.commit().unwrap();
        assert!(store.Users.get(OWNER, &1).is_none());
    }

    #[test]
    fn txn_index_iter_mut_updates_index_on_field_change() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        txn.Users
            .indexes()
            .name
            .iter_mut()
            .for_each(|(_, _, v)| v.name = "Charlie".to_owned());
        assert!(txn.Users.indexes().name.get("Alice").is_none());
        assert_eq!(
            txn.Users.indexes().name.get("Charlie").unwrap(),
            (
                1,
                Row {
                    name: "Charlie".to_owned(),
                    age: 0
                }
            )
        );
    }

    // Multi-row transactional index mutation: every row must be re-indexed under its new key.
    #[test]
    fn txn_index_iter_mut_updates_multiple_rows() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        txn.Users.insert(2, row("Bob"));
        txn.Users
            .indexes()
            .name
            .iter_mut()
            .for_each(|(_, _, v)| v.name.push('!'));

        assert!(txn.Users.indexes().name.get("Alice").is_none());
        assert!(txn.Users.indexes().name.get("Bob").is_none());
        assert_eq!(
            txn.Users.indexes().name.get("Alice!").unwrap(),
            (1, row("Alice!"))
        );
        assert_eq!(
            txn.Users.indexes().name.get("Bob!").unwrap(),
            (2, row("Bob!"))
        );
    }

    // A row visited by the index iterator but left unmodified must remain correctly indexed.
    #[test]
    fn txn_index_iter_mut_visited_unmodified_row_stays_indexed() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        txn.Users.insert(2, row("Bob"));
        txn.Users
            .indexes()
            .name
            .iter_mut()
            .for_each(|(_, base_key, v)| {
                if *base_key == 1 {
                    v.name = "Zara".to_owned();
                }
            });

        assert!(txn.Users.indexes().name.get("Alice").is_none());
        assert_eq!(
            txn.Users.indexes().name.get("Zara").unwrap(),
            (1, row("Zara"))
        );
        assert_eq!(
            txn.Users.indexes().name.get("Bob").unwrap(),
            (2, row("Bob"))
        );
    }

    #[test]
    fn txn_base_insert_updates_index() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        assert_eq!(
            txn.Users.indexes().name.get("Alice").unwrap(),
            (1, row("Alice"))
        );
    }

    #[test]
    fn txn_base_mutate_updates_index_on_field_change() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        txn.Users.with_mut(&1, |v| v.name = "Bob".to_owned());
        assert!(txn.Users.indexes().name.get("Alice").is_none());
        assert_eq!(
            txn.Users.indexes().name.get("Bob").unwrap(),
            (
                1,
                Row {
                    name: "Bob".to_owned(),
                    age: 0
                }
            )
        );
    }

    #[test]
    fn txn_base_remove_updates_index() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        txn.Users.remove(&1);
        assert!(txn.Users.indexes().name.get("Alice").is_none());
    }

    #[test]
    fn txn_base_clear_updates_index() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        txn.Users.insert(2, row("Bob"));
        txn.Users.clear();
        assert!(txn.Users.indexes().name.get("Alice").is_none());
        assert!(txn.Users.indexes().name.get("Bob").is_none());
    }

    #[test]
    fn txn_base_iter_mut_updates_index_on_field_change() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        txn.Users.iter_mut().next().unwrap().1.name = "Charlie".to_owned();
        assert!(txn.Users.indexes().name.get("Alice").is_none());
        assert_eq!(
            txn.Users.indexes().name.get("Charlie").unwrap(),
            (
                1,
                Row {
                    name: "Charlie".to_owned(),
                    age: 0
                }
            )
        );
    }

    // Rows inserted after a `clear()` in the same transaction live in the delete
    // mask's pending map, not in `data`. Iterating the base table mutably must still leave those
    // rows correctly indexed.
    #[test]
    fn txn_base_iter_mut_after_clear_keeps_new_rows_indexed() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        txn.Users.clear();
        txn.Users.insert(2, row("Bob"));
        for (_, v) in txn.Users.iter_mut() {
            v.name.push('!');
        }
        txn.commit().unwrap();

        let index = store.with_owner(OWNER).Users.indexes().name;
        assert!(index.get("Alice").is_none());
        assert!(index.get("Bob").is_none());
        assert_eq!(index.get("Bob!").unwrap(), (2, row("Bob!")));
    }

    #[test]
    fn txn_base_iter_mut_after_clear_unmodified_stays_indexed() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.clear();
        txn.Users.insert(2, row("Bob"));
        for (_, _v) in txn.Users.iter_mut() {}
        txn.commit().unwrap();

        let index = store.with_owner(OWNER).Users.indexes().name;
        assert_eq!(index.get("Bob").unwrap(), (2, row("Bob")));
    }

    // Removing a key then iterating mutably: the removed row must not be re-indexed, and a surviving
    // row mutated through the iterator must be re-indexed under its new key.
    #[test]
    fn txn_base_iter_mut_after_remove_keeps_index_consistent() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice"));
        txn.Users.insert(2, row("Bob"));
        txn.Users.remove(&1);
        for (_, v) in txn.Users.iter_mut() {
            v.name.push('!');
        }
        txn.commit().unwrap();

        let index = store.with_owner(OWNER).Users.indexes().name;
        assert!(index.get("Alice").is_none());
        assert!(index.get("Alice!").is_none());
        assert!(index.get("Bob").is_none());
        assert_eq!(index.get("Bob!").unwrap(), (2, row("Bob!")));
    }

    #[test]
    fn ro_txn_index_get_returns_none_when_absent() {
        let store = KvStore::new();
        let txn = store.begin_ro_transaction(OWNER);
        assert!(txn.Users.indexes().name.get("Alice").is_none());
    }

    #[test]
    fn ro_txn_index_get_returns_value_inserted_before_txn() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        let txn = store.begin_ro_transaction(OWNER);
        assert_eq!(
            txn.Users.indexes().name.get("Alice").unwrap(),
            (1, row("Alice"))
        );
    }

    #[test]
    fn ro_txn_index_with_returns_some() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        let txn = store.begin_ro_transaction(OWNER);
        assert_eq!(
            txn.Users
                .indexes()
                .name
                .with("Alice", |k, v| {
                    assert_eq!(*k, 1);
                    v.name.len()
                })
                .unwrap(),
            5
        );
    }

    #[test]
    fn ro_txn_index_iter_cloned_yields_rows() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        store.Users.insert(OWNER, 2, row("Bob"));
        let txn = store.begin_ro_transaction(OWNER);

        let table = txn.Users.indexes().name;
        let mut rows: Vec<_> = table.iter().collect();
        rows.sort_by(|a, b| a.0.cmp(b.0));
        assert_eq!(
            rows,
            vec![
                (&"Alice".to_owned(), &1, &row("Alice")),
                (&"Bob".to_owned(), &2, &row("Bob"))
            ]
        );
    }

    #[test]
    fn ro_txn_index_for_each_yields_rows() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        store.Users.insert(OWNER, 2, row("Bob"));
        let txn = store.begin_ro_transaction(OWNER);
        let mut names: Vec<String> = Vec::new();
        txn.Users
            .indexes()
            .name
            .iter()
            .for_each(|(k, ..)| names.push(k.clone()));
        names.sort();
        assert_eq!(names, vec!["Alice".to_owned(), "Bob".to_owned()]);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "Ownership violation"))]
    fn txn_index_remove_wrong_owner_panics() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OTHER);
        txn.Users.indexes().name.remove("Alice");
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "Ownership violation"))]
    fn txn_index_iter_mut_wrong_owner_panics() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OTHER);
        let mut table = txn.Users.indexes().name;
        let _iter = table.iter_mut();
    }

    #[test]
    fn txn_index_base_mutate_indexed_field_rolled_back() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        {
            let mut txn = store.begin_transaction(OWNER);
            txn.Users
                .indexes()
                .name
                .with_mut("Alice", |_, v| v.name = "Bob".to_owned())
                .unwrap();
        }
        assert_eq!(
            store.Users.indexes().name.get(OWNER, "Alice").unwrap(),
            (1, row("Alice"))
        );
        assert!(store.Users.indexes().name.get(OWNER, "Bob").is_none());
        assert_eq!(store.Users.get(OWNER, &1), Some(row("Alice")));
    }

    #[test]
    fn txn_index_remove_rolled_back_on_drop() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));
        {
            let mut txn = store.begin_transaction(OWNER);
            txn.Users.indexes().name.remove("Alice");
        }
        assert_eq!(
            store.Users.indexes().name.get(OWNER, "Alice").unwrap(),
            (1, row("Alice"))
        );
        assert_eq!(store.Users.get(OWNER, &1), Some(row("Alice")));
    }

    #[test]
    fn raw_base_then_txn_insert_commit_both_visible() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice"));

        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(2, row("Bob"));
        txn.commit().unwrap();

        assert_eq!(
            store.Users.indexes().name.get(OWNER, "Alice").unwrap(),
            (1, row("Alice"))
        );
        assert_eq!(
            store.Users.indexes().name.get(OWNER, "Bob").unwrap(),
            (2, row("Bob"))
        );
        assert_eq!(store.Users.get(OWNER, &1), Some(row("Alice")));
        assert_eq!(store.Users.get(OWNER, &2), Some(row("Bob")));
    }
}

#[cfg(test)]
mod test_poison {
    use crate::{Error, KvErrorExt, store};

    #[derive(Clone, Debug, PartialEq)]
    pub struct Row {
        pub name: String,
        pub email: String,
    }

    fn row(name: &str, email: &str) -> Row {
        Row {
            name: name.to_owned(),
            email: email.to_owned(),
        }
    }

    store!(
        tables: {
            Users(u32 => Row; OWNER; index(name: String); index(email: String)),
            AssertingUsers(u32 => Row; OWNER; index(name: String; assert_unique)),
        }
    );

    const OWNER: &str = "owner";

    #[test]
    fn ops_return_error_when_poisoned() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice", "alice1@x.com"));
        txn.Users.insert(2, row("Alice", "alice2@x.com"));

        let mut index_name = txn.Users.indexes().name;
        assert_eq!(
            index_name.check_consistent(),
            Err(Error::NonUniqueIndexKey("Users by name"))
        );

        let result = index_name.get("Alice");
        assert!(matches!(result, Err(Error::NonUniqueIndexKey(_))));

        // The `panic`s ensure that the closure is not called.
        let result = index_name.with("Alice", |_, _| panic!());
        assert!(matches!(result, Err(Error::NonUniqueIndexKey(_))));

        let result = index_name.with_mut("Alice", |_, _| panic!());
        assert!(matches!(result, Err(Error::NonUniqueIndexKey(_))));
    }

    #[test]
    fn check_consistent_ok_when_unique() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice", "alice1@x.com"));
        assert!(store.Users.indexes().name.check_consistent().is_ok());
    }

    #[test]
    fn base_table_unaffected_when_index_poisoned() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice", "alice1@x.com"));
        txn.Users.insert(2, row("Alice", "alice2@x.com"));

        // The `name` index is poisoned, but the base table can still be read by primary key.
        assert_eq!(txn.Users.get(&1), Some(row("Alice", "alice1@x.com")));
        assert_eq!(txn.Users.get(&2), Some(row("Alice", "alice2@x.com")));
    }

    #[test]
    fn sibling_index_not_poisoned() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice", "alice1@x.com"));
        txn.Users.insert(2, row("Alice", "alice2@x.com"));

        // `name` is poisoned (both "Alice")...
        assert!(matches!(
            txn.Users.indexes().name.check_consistent(),
            Err(Error::NonUniqueIndexKey(_))
        ));
        // ...but `email` has distinct keys and stays consistent.
        let email_index = txn.Users.indexes().email;
        assert!(email_index.check_consistent().is_ok());
        assert_eq!(
            email_index.get("alice1@x.com").unwrap(),
            (1, row("Alice", "alice1@x.com"))
        );
        assert_eq!(
            email_index.get("alice2@x.com").unwrap(),
            (2, row("Alice", "alice2@x.com"))
        );
    }

    #[test]
    fn clear_unpoisons() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice", "alice1@x.com"));
        txn.Users.insert(2, row("Alice", "alice2@x.com"));

        assert!(txn.Users.indexes().name.check_consistent().is_err());

        txn.Users.clear();
        let index = txn.Users.indexes().name;
        assert!(index.check_consistent().is_ok());
        assert!(index.get("Alice").is_none());
    }

    #[test]
    fn txn_get_returns_error_when_poisoned() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice", "alice1@x.com"));
        txn.Users.insert(2, row("Alice", "alice2@x.com"));

        assert!(matches!(
            txn.Users.indexes().name.get("Alice"),
            Err(Error::NonUniqueIndexKey(_))
        ));
    }

    #[test]
    fn txn_check_consistent_errors_when_poisoned() {
        let store = KvStore::new();
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(1, row("Alice", "alice1@x.com"));
        txn.Users.insert(2, row("Alice", "alice2@x.com"));

        assert!(matches!(
            txn.Users.indexes().name.check_consistent(),
            Err(Error::NonUniqueIndexKey(_))
        ));
    }

    #[test]
    fn txn_commit_fails_when_index_poisoned() {
        let store = KvStore::new();
        {
            let mut txn = store.begin_transaction(OWNER);
            txn.Users.insert(1, row("Alice", "alice1@x.com"));
            txn.Users.insert(2, row("Alice", "alice2@x.com"));
            assert!(matches!(txn.commit(), Err(Error::NonUniqueIndexKey(_))));
        }

        // The failed commit rolled everything back: the store is clean and consistent.
        assert!(store.Users.get(OWNER, &1).is_none());
        assert!(store.Users.get(OWNER, &2).is_none());
        assert!(store.Users.indexes().name.check_consistent().is_ok());
    }

    #[test]
    fn txn_poison_rolled_back_on_drop() {
        let store = KvStore::new();
        {
            let mut txn = store.begin_transaction(OWNER);
            txn.Users.insert(1, row("Alice", "alice1@x.com"));
            txn.Users.insert(2, row("Alice", "alice2@x.com"));
            // dropped without committing
        }

        let index = store.with_owner(OWNER).Users.indexes().name;
        assert!(index.check_consistent().is_ok());
        assert!(index.get("Alice").is_none());
        assert!(store.Users.is_empty());
    }

    #[test]
    fn txn_poison_against_committed_rolled_back() {
        let store = KvStore::new();
        // Commit Alice(1) first, so the "Alice" name-index entry is already committed.
        store.Users.insert(OWNER, 1, row("Alice", "alice1@x.com"));

        {
            // This txn collides with the *already-committed* "Alice" index entry, so the index is
            // poisoned without it ever being recorded as `modified` within the txn.
            let mut txn = store.begin_transaction(OWNER);
            txn.Users.insert(2, row("Alice", "alice2@x.com"));
            // rollback on drop
        }

        // An unrelated, valid commit must succeed — the rolled-back poison must not leak into it.
        let mut txn = store.begin_transaction(OWNER);
        txn.Users.insert(3, row("Bob", "bob@x.com"));
        txn.commit().unwrap();

        let index = store.with_owner(OWNER).Users.indexes().name;
        assert!(
            index.check_consistent().is_ok(),
            "index should not be poisoned after the colliding txn was rolled back"
        );
        assert_eq!(
            index.get("Alice").unwrap(),
            (1, row("Alice", "alice1@x.com"))
        );
        assert_eq!(index.get("Bob").unwrap(), (3, row("Bob", "bob@x.com")));
    }

    #[test]
    fn assert_unique_distinct_keys_ok() {
        let store = KvStore::new();
        store
            .AssertingUsers
            .insert(OWNER, 1, row("Alice", "alice1@x.com"));
        store
            .AssertingUsers
            .insert(OWNER, 2, row("Bob", "bob@x.com"));
        assert_eq!(
            store
                .AssertingUsers
                .indexes()
                .name
                .get(OWNER, "Alice")
                .unwrap(),
            (1, row("Alice", "alice1@x.com"))
        );
    }

    #[test]
    #[should_panic(expected = "non-unique")]
    fn assert_unique_duplicate_base_insert_panics() {
        let store = KvStore::new();
        store
            .AssertingUsers
            .insert(OWNER, 1, row("Alice", "alice1@x.com"));
        store
            .AssertingUsers
            .insert(OWNER, 2, row("Alice", "alice2@x.com"));
    }

    #[test]
    fn raw_base_try_insert_duplicate_errors_and_rolls_back() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice", "alice1@x.com"));

        // The duplicate "Alice" name-index key makes the insert fail.
        assert!(matches!(
            store
                .Users
                .try_insert(OWNER, 2, row("Alice", "alice2@x.com")),
            Err(Error::NonUniqueIndexKey(_))
        ));

        // The failed insert rolled back: the index is consistent and only the original row remains.
        let index = store.with_owner(OWNER).Users.indexes().name;
        assert!(index.check_consistent().is_ok());
        assert_eq!(
            index.get("Alice").unwrap(),
            (1, row("Alice", "alice1@x.com"))
        );
        assert_eq!(
            store.Users.get(OWNER, &1),
            Some(row("Alice", "alice1@x.com"))
        );
        assert_eq!(store.Users.get(OWNER, &2), None);
    }

    #[test]
    fn raw_base_insert_duplicate_panics_and_rolls_back() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice", "alice1@x.com"));

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            store.Users.insert(OWNER, 2, row("Alice", "alice2@x.com"));
        }));
        assert!(result.is_err(), "duplicate raw insert should panic");

        // The panicked insert has been rolled back, leaving the store consistent.
        let index = store.with_owner(OWNER).Users.indexes().name;
        assert!(index.check_consistent().is_ok());
        assert_eq!(
            index.get("Alice").unwrap(),
            (1, row("Alice", "alice1@x.com"))
        );
        assert_eq!(store.Users.get(OWNER, &2), None);
    }

    #[test]
    fn raw_with_mut_collision_returns_error_and_rolls_back() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice", "alice1@x.com"));
        store.Users.insert(OWNER, 2, row("Bob", "bob@x.com"));

        // Renaming Bob to "Alice" collides on the `name` index, so the mini-transaction fails.
        assert!(matches!(
            store
                .Users
                .with_mut(OWNER, &2, |r| r.name = "Alice".to_owned()),
            Err(Error::NonUniqueIndexKey(_))
        ));

        // Rolled back: Bob is unchanged and the index is consistent.
        assert_eq!(store.Users.get(OWNER, &2), Some(row("Bob", "bob@x.com")));
        let index = store.with_owner(OWNER).Users.indexes().name;
        assert!(index.check_consistent().is_ok());
        assert_eq!(index.get("Bob").unwrap(), (2, row("Bob", "bob@x.com")));
        assert_eq!(
            index.get("Alice").unwrap(),
            (1, row("Alice", "alice1@x.com"))
        );
    }

    #[test]
    fn raw_with_mut_panic_rolls_back() {
        let store = KvStore::new();
        store.Users.insert(OWNER, 1, row("Alice", "alice1@x.com"));

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = store.Users.with_mut(OWNER, &1, |r| {
                r.name = "Zelda".to_owned();
                panic!("boom");
            });
        }));
        assert!(result.is_err(), "panicking closure should propagate");

        // The mutation (and its index update) rolled back; "Alice" is intact and queryable.
        assert_eq!(
            store.Users.get(OWNER, &1),
            Some(row("Alice", "alice1@x.com"))
        );
        let index = store.with_owner(OWNER).Users.indexes().name;
        assert!(index.check_consistent().is_ok());
        assert_eq!(
            index.get("Alice").unwrap(),
            (1, row("Alice", "alice1@x.com"))
        );
        assert!(index.get("Zelda").is_none());
    }
}
