//! Iterate over a table

use std::{hash::Hash, marker::PhantomData};

use crate::{
    schema::{IndexDesc, TableDesc},
    storage::{Table, TableIterator as InnerIterator},
    transactions::TxnId,
};

/// Phantom type to iterate over keys.
#[doc(hidden)]
pub struct Keys;
/// Phantom type to iterate over Values.
#[doc(hidden)]
pub struct Values;
/// Phantom type to iterate over key/value pairs.
#[doc(hidden)]
pub struct KeysAndValues;

type Indexes<D> =
    Table<<D as IndexDesc>::BaseTable, <<D as IndexDesc>::BaseTable as TableDesc>::IndexStorage>;

/// An iterator for a single table (described by the generic parameter `D`) in the KV store.
///
/// This is basically just a wrapper for an iterator over the `HashMap` representing the table. The
/// lifetime `'a` is that of the borrow of the table; whoever creates the iterator is responsible
/// for ensuring that the table is locked (and not mutated) for that lifetime.
pub struct TableIterator<'a, D: TableDesc, Kind> {
    inner: InnerIterator<'a, D>,
    _kind: PhantomData<Kind>,
}

impl<'a, D: TableDesc, Kind> TableIterator<'a, D, Kind> {
    pub(crate) fn new(inner: InnerIterator<'a, D>) -> Self {
        TableIterator {
            inner,
            _kind: PhantomData,
        }
    }
}

impl<'a, D: TableDesc> Iterator for TableIterator<'a, D, KeysAndValues> {
    type Item = (&'a D::Key, &'a D::Value);

    fn next(&mut self) -> Option<Self::Item> {
        // Iterate by delegating to the `HashMap` iterator.
        self.inner.next()
    }
}

impl<'a, D: TableDesc> Iterator for TableIterator<'a, D, Keys> {
    type Item = &'a D::Key;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }
}

impl<'a, D: TableDesc> Iterator for TableIterator<'a, D, Values> {
    type Item = &'a D::Value;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }
}

/// An iterator for an indexed table (described by the generic parameter `D`) in the KV
/// store.
///
/// As for [`TableIterator`], `'a` is the lifetime of the borrow of the (base) table.
pub struct IndexIterator<'a, D: IndexDesc, Kind> {
    base: &'a Indexes<D>,
    /// An iterator over the `HashMap` representing the index.
    inner: InnerIterator<'a, D>,
    txn_id: TxnId,
    _kind: PhantomData<Kind>,
}

impl<'a, D: IndexDesc, Kind> IndexIterator<'a, D, Kind> {
    pub(crate) fn new(base: &'a Indexes<D>, inner: InnerIterator<'a, D>, txn_id: TxnId) -> Self {
        IndexIterator {
            base,
            inner,
            txn_id,
            _kind: PhantomData,
        }
    }
}

impl<'a, D: IndexDesc> Iterator for IndexIterator<'a, D, KeysAndValues>
where
    D::Value: Hash + Eq,
{
    type Item = (
        &'a D::Key,
        &'a <D::BaseTable as TableDesc>::Key,
        &'a <D::BaseTable as TableDesc>::Value,
    );

    fn next(&mut self) -> Option<Self::Item> {
        let (k, bk) = self.inner.next()?;
        let value = self.base.get(bk, self.txn_id)?;

        Some((k, bk, value))
    }
}

impl<'a, D: IndexDesc> Iterator for IndexIterator<'a, D, Keys> {
    type Item = &'a D::Key;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }
}
