//! Traits and macros for defining the KvStore schema.

use std::{any::Any, hash::Hash, ops::Deref};

use crate::{
    Owner, SingletonTransaction, TableTransaction,
    pub_sub::Subscriptions,
    storage::{Table, VersionedValue},
    transactions::TxnId,
};

/// A singleton key/value.
///
/// Prefer to use the macros in this module rather than this trait directly.
pub trait SingletonDesc: Sized + 'static {
    /// The datum's owner.
    const OWNER: Owner;

    /// The type of the value.
    type Value: Any + Send + Sync;
    /// The type of the notification for this singleton (either `Self::Value` or `()`).
    type NotificationValue: Clone;
    /// The storage for this singleton KV.
    type Storage: GeneratedStorage;

    /// Get a clone of the value from `storage` if the singleton uses cloning for notification values,
    /// or `()` if not.
    fn get_cloned(storage: &Self::Storage, txn_id: TxnId) -> Option<Self::NotificationValue>;

    /// Get a reference to the field storing this singleton in `storage`.
    fn get_ref(storage: &Self::Storage) -> &VersionedValue<Option<Self::Value>>;

    /// Get a mutable reference to the field storing this singleton in `storage`.
    fn get_mut(storage: &mut Self::Storage) -> &mut VersionedValue<Option<Self::Value>>;

    /// Get a pointer to the field storing this singleton in the storage pointed to by `storage`.
    ///
    /// Unlike `get_mut`, this does not create a reference to the whole storage, and so can be used
    /// while there are live references to other parts of the storage.
    ///
    /// Implementations must return a pointer to a field of `*storage` which is distinct from the
    /// fields used for every other singleton and table in the storage.
    ///
    /// # Safety
    ///
    /// `storage` must be valid for reads and writes (it need not be uniquely borrowed).
    unsafe fn value_ptr(storage: *mut Self::Storage) -> *mut VersionedValue<Option<Self::Value>>;

    /// Convert an optional reference to this singleton's value to it's notification value.
    fn notif_value(value: &Option<Self::Value>) -> &Option<Self::NotificationValue>;

    /// Create a notification from an event.
    fn make_notification(
        event: crate::SingletonEvent<Self, Self::NotificationValue>,
    ) -> <Self::Storage as GeneratedStorage>::Notification;

    /// Create a transactinal accessor of this singleton, existing within `txn`.
    fn make_txn_view<'a, 'b>(
        txn: &'b mut <Self::Storage as GeneratedStorage>::Transaction<'a>,
    ) -> &'b mut SingletonTransaction<'a, Self::Storage, Self>;
}

/// Describes tabular key/values in the store.
///
/// Prefer to use the macros in this module rather than this trait directly.
pub trait TableDesc: Sized + 'static {
    /// The name of the table.
    const NAME: &'static str;
    /// The table's owner.
    const OWNER: Owner;

    /// The type of the key.
    type Key: Hash + Eq + Clone;
    /// The type of the value.
    type Value: Any + Send + Sync;
    /// The storage for the table.
    type Storage: GeneratedStorage;
    /// The storage type for keeping this table's indexes.
    type IndexStorage: IndexStorage<Self::Key, Self::Value>;

    /// Get a reference to the table in storage.
    fn get_table(storage: &Self::Storage) -> &Table<Self, Self::IndexStorage>;

    /// Get a pointer to the table in the storage pointed to by `storage`.
    ///
    /// This does not create a reference to the whole storage, and so can be used while there are
    /// live references to other parts of the storage.
    ///
    /// Implementations must return a pointer to a field of `*storage` which is distinct from the
    /// fields used for every other table and singleton in the storage.
    ///
    /// # Safety
    ///
    /// `storage` must be valid for reads and writes (it need not be uniquely borrowed).
    unsafe fn table_ptr(storage: *mut Self::Storage) -> *mut Table<Self, Self::IndexStorage>;

    /// Compare two references to this table's value type, returns `true` if the value type impls
    /// `PartialEq` and the values are equal. **Panics** if `Self::Value` does not impl `PartialEq`.
    fn value_eq(a: &Self::Value, b: &Self::Value) -> bool;

    /// Create a transactinal accessor of this table, existing within `txn`.
    fn make_txn_view<'a, 'b>(
        txn: &'b mut <Self::Storage as GeneratedStorage>::Transaction<'a>,
    ) -> &'b mut TableTransaction<'a, Self::Storage, Self>;
}

/// A table where changes can generate notifications to subscribers.
pub trait Notifiable: TableDesc {
    /// The type of the notification for this table (either `Self::Value` or `()`).
    type NotificationValue: Clone;

    /// Create a notification from an event.
    fn make_notification(
        event: crate::Event<Self, Self::Key, Self::NotificationValue>,
    ) -> <Self::Storage as GeneratedStorage>::Notification;

    /// Create a value for a notification, possibly by cloning `value`.
    fn clone_value_for_notification(value: &Self::Value) -> Self::NotificationValue;
}

/// A table with indexes.
pub trait Indexable: TableDesc {
    /// The macro-generated struct with a field for each of the table's indexes, see
    /// [`crate::Table::indexes`].
    type Indexes<'store>: From<&'store crate::KvStore<Self::Storage>>;

    /// The owner-carrying counterpart of [`Self::Indexes`], see
    /// [`crate::TableWithOwner::indexes`].
    type IndexesWithOwner<'store>: From<(&'store crate::KvStore<Self::Storage>, Owner)>;

    /// The transactional counterpart of [`Self::Indexes`], see
    /// [`crate::TableTransaction::indexes`].
    type TransactionIndexes<'guard, 'txn>: From<*mut crate::Transaction<'guard, Self::Storage>>;

    /// The read-only transactional counterpart of [`Self::Indexes`], see
    /// [`crate::RoTableTransaction::indexes`].
    type RoTransactionIndexes<'guard, 'txn>: From<
        *const crate::RoTransaction<'guard, Self::Storage>,
    >;
}

/// Describes a table used as an index.
///
/// An index table is stored within its base table (in the base table's `indexes` field). The index
/// table's values are keys of the base table.
pub trait IndexDesc: TableDesc<Value: Hash + Eq + Clone> {
    /// The table which is indexed.
    type BaseTable: Notifiable + TableDesc<Storage = Self::Storage, Key = Self::Value>;

    /// Get a reference to this index table from its base table.
    fn index(
        base: &Table<Self::BaseTable, <Self::BaseTable as TableDesc>::IndexStorage>,
    ) -> &Table<Self, Self::IndexStorage>;

    /// Get a mutable reference to this index table from its base table.
    fn index_mut(
        base: &mut Table<Self::BaseTable, <Self::BaseTable as TableDesc>::IndexStorage>,
    ) -> &mut Table<Self, Self::IndexStorage>;
}

/// Operations on an index.
pub trait IndexStorage<K: Hash + Eq, V: Any + Send + Sync>: Default {
    /// Clear the whole index.
    fn clear(
        &mut self,
        txn_id: crate::transactions::TxnId,
        max_committed_id: crate::transactions::TxnId,
    );

    /// An item has been inserted into the index.
    fn on_insert<Q>(
        &mut self,
        key: &Q,
        value: &V,
        txn_id: crate::transactions::TxnId,
        max_committed_id: crate::transactions::TxnId,
    ) where
        K: std::borrow::Borrow<Q>,
        Q: ?Sized + std::hash::Hash + Eq + std::borrow::ToOwned<Owned = K>;

    /// An item has been removed from the index.
    fn on_remove(
        &mut self,
        value: &V,
        txn_id: crate::transactions::TxnId,
        max_committed_id: crate::transactions::TxnId,
    );
}

impl<K: Hash + Eq, V: Any + Send + Sync> IndexStorage<K, V> for () {
    fn clear(
        &mut self,
        _txn_id: crate::transactions::TxnId,
        _max_committed_id: crate::transactions::TxnId,
    ) {
    }

    fn on_insert<Q>(
        &mut self,
        _key: &Q,
        _value: &V,
        _id: crate::transactions::TxnId,
        _max_committed_id: crate::transactions::TxnId,
    ) where
        K: std::borrow::Borrow<Q>,
        Q: ?Sized + std::hash::Hash + Eq,
    {
    }

    fn on_remove(
        &mut self,
        _value: &V,
        _txn_id: crate::transactions::TxnId,
        _max_committed_id: crate::transactions::TxnId,
    ) {
    }
}

/// A storage implementation.
///
/// This should be considered a sealed trait and not implemented except by the macros in this module.
/// Unfortunately it has to be public because of macro visibility hygiene.
#[doc(hidden)]
pub trait GeneratedStorage: Default + Send + Sync + 'static {
    /// An enum of all notification types.
    type Notification: Clone + Send + 'static;
    /// The type of transactions for a store.
    ///
    /// Will contain a field for each singleton and table.
    type Transaction<'a>: crate::transactions::SchemaTransaction
    where
        Self: 'a;
    /// A read-only version of `Transaction`.
    type RoTransaction<'a>: crate::transactions::SchemaTransaction
    where
        Self: 'a;

    /// Commit a transaction by applying all tables' transaction state to their permanent data.
    ///
    /// This operation must be atomic. I.e., it will only fail without any tables committed, and if it
    /// succeeds, then all masks have committed.
    fn commit_txn(
        &mut self,
        txn_id: TxnId,
        notifications: &mut crate::Notifications<Self::Notification>,
        subscriptions: &Subscriptions<Self::Notification>,
    ) -> crate::Result<()>;

    /// Delete any uncommitted per-transaction state associated with `txn_id` held in tables.
    fn gc_txn(&mut self, txn_id: TxnId);

    /// Create a schema-specific transaction from a general transaction object.
    fn make_txn<'a>(store_txn: crate::Transaction<'a, Self>) -> Self::Transaction<'a>;

    /// Create a schema-specific read-only transaction from a general transaction object.
    fn make_ro_txn<'a>(store_txn: crate::RoTransaction<'a, Self>) -> Self::RoTransaction<'a>;
}

/// Implemented by stores generated by the schma macros.
pub trait GeneratedStore<Storage: GeneratedStorage>:
    'static
    + Send
    + Sync
    + Deref<Target = crate::KvStore<Storage>>
    + From<
        std::sync::Weak<
            dyn crate::Notifier<Notification = <Storage as GeneratedStorage>::Notification>,
        >,
    >
{
}

/// Declare the schema of a key/value store. Generates the store itself with the specified tables and
/// singletons.
///
/// The syntax is:
/// ```ignore
/// store!(
///   kvs: { Name(ValueType; owner; notify(None|Clone)?),* }
///   tables: { Name(KeyType => ValueType; owner; indexes?; notify(None|Clone)?),* }
/// )
/// ```
/// where `Name` is an identifier to name the table or singleton (in which case it is also the key),
/// `KeyType` and `ValueType` are types. `owner` is an expression which evaluates to an `Owner`.
/// `Name` is used as a type argument to KvStore methods to identify the table or singleton.
///
/// # Example:
///
/// ```rust
/// # use ts_kv_store::store;
/// # use std::sync::Arc;
/// # const GRAPH_OWNER: ts_kv_store::Owner = "foo";
/// # const NODES_OWNER: ts_kv_store::Owner = "bar";
/// # const EDGES_OWNER: ts_kv_store::Owner = "baz";
/// # pub struct Node;
/// # #[derive(Clone, PartialEq)]
/// # pub struct Gid;
/// # pub trait Edge {}
/// store!(
///   kvs: {
///     GraphId(Arc<Gid>; GRAPH_OWNER),
///   }
///   tables: {
///     Nodes(&'static str => Node; NODES_OWNER),
///     Edges(u32 => Box<dyn Edge + Send + Sync>; EDGES_OWNER),
///   }
/// );
/// ```
///
/// # Indexes
///
/// The syntax of an index is `index(field: Type(; assert_unique)?)` where `field` is the name of
/// a field in the value type of the base table and `Type` is the type of that field. You can
/// specify multiple indexes for each table, separated with a semicolon. E.g.,
///
/// ```rust
/// # use ts_kv_store::store;
/// # const NODES_OWNER: ts_kv_store::Owner = "foo";
/// # pub struct Node { a: u32, b: String };
/// store!(
///   tables: {
///     Nodes(
///       &'static str => Node;
///       NODES_OWNER;
///       index(a: u32);
///       index(b: String; assert_unique);
///       index(c: String = |node: &Node| [format!("{}-{}", node.a, node.b)]);
///     )
///   }
/// );
/// ```
///
/// This will create indexes on nodes for fields `a`, `b`, and `c`. `c`'s index key is
/// computed by the specified closure, which is expected to return
/// `impl IntoIterator<Item = I>` (where `I` is the index key type), for example an `Option<I>` for
/// zero or one index keys per value.
///
/// Index fields must uniquely identify a row in the base table. If multiple rows in the base table
/// have the same key in the index, then by default the index will be 'poisoned' and accessing the
/// index or trying to commit a transaction where an index is poisoned will return an error (`NonUniqueIndexKey`).
/// By adding `assert_unique` to an index declaration (after the index field, separated with a semicolon),
/// attempting to store multiple rows with the same index key will cause a panic.
///
/// ## Notifications
///
/// The `notify(...)` argument is used to control notifications about the table or singleton. Accepted
/// values are `Clone` and `None`. The default is `None`.
///
/// The `None` behaviour is that only keys are sent in notifications. The `Clone` behaviour is
/// that values are cloned and included in notifications. This requires that the value type implements
/// the `Clone` trait.
#[macro_export]
macro_rules! store {
    (
        $(kvs: { $($sname:ident($svalue_ty:ty; $sowner:expr $(; notify($snotif:ident))?)),* $(,)? })?
        $(tables: { $(
            $name:ident (
                $key_ty:ty => $value_ty:ty;
                $owner:expr
                $(; index($field:ident: $field_ty:ty $(= $get_idx:expr)? $(; $unique:ident)?))*
                $(; notify($notif:ident))?
                $(;)?
            )
        ),* $(,)? })?
    ) => {
        $($(
            /// Describes a singleton in the KV store.
            #[allow(non_camel_case_types)]
            pub struct $sname;

            impl $crate::schema::SingletonDesc for $sname {
                const OWNER: $crate::Owner = $sowner;
                type Value = $svalue_ty;
                type NotificationValue = $crate::notification_value_type!($svalue_ty $(; notify($snotif))?);
                type Storage = TableStorage;

                fn get_cloned(_storage: &Self::Storage, _txn_id: $crate::transactions::TxnId) -> Option<Self::NotificationValue> {
                    $crate::get_cloned_notification_value!($sname, _storage, _txn_id $(; notify($snotif))?)
                }

                fn get_ref(storage: &Self::Storage) -> &$crate::storage::VersionedValue<Option<Self::Value>> {
                    &storage.$sname
                }

                fn get_mut(storage: &mut Self::Storage) -> &mut $crate::storage::VersionedValue<Option<Self::Value>>{
                    &mut storage.$sname
                }

                unsafe fn value_ptr(storage: *mut Self::Storage) -> *mut $crate::storage::VersionedValue<Option<Self::Value>> {
                    // SAFETY: `storage` is valid by the precondition of `value_ptr`; only a raw
                    // pointer is created.
                    unsafe { &raw mut (*storage).$sname }
                }

                fn notif_value(_value: &Option<Self::Value>) -> &Option<Self::NotificationValue> {
                    $crate::notification_value!(_value $(; notify($snotif))? )
                }

                fn make_notification(
                    event: $crate::SingletonEvent<Self, Self::NotificationValue>,
                ) -> <Self::Storage as $crate::schema::GeneratedStorage>::Notification {
                    Notification::$sname(event)
                }

                fn make_txn_view<'a, 'b>(txn: &'b mut <Self::Storage as $crate::schema::GeneratedStorage>::Transaction<'a>) -> &'b mut $crate::SingletonTransaction<'a, Self::Storage, Self> {
                    &mut txn.$sname
                }
            }
        )*)?
        $($(
            /// Describes a table in the KV store.
            #[derive(Default)]
            pub struct $name;

            impl $crate::schema::TableDesc for $name {
                const NAME: &'static str = stringify!($name);
                const OWNER: $crate::Owner = $owner;
                type Key = $key_ty;
                type Value = $value_ty;
                type Storage = TableStorage;
                type IndexStorage = index::$name::Storage;

                fn get_table(storage: &TableStorage) -> &$crate::storage::Table<Self, Self::IndexStorage> {
                    &storage.$name
                }
                unsafe fn table_ptr(storage: *mut TableStorage) -> *mut $crate::storage::Table<Self, Self::IndexStorage> {
                    // SAFETY: `storage` is valid by the precondition of `table_ptr`; only a raw
                    // pointer is created.
                    unsafe { &raw mut (*storage).$name }
                }
                fn make_txn_view<'a, 'b>(txn: &'b mut <Self::Storage as $crate::schema::GeneratedStorage>::Transaction<'a>) -> &'b mut $crate::TableTransaction<'a, Self::Storage, Self> {
                    &mut txn.$name
                }

                $crate::value_eq!(Self::Value);
            }

            impl $crate::schema::Notifiable for $name {
                type NotificationValue = $crate::notification_value_type!($value_ty $(; notify($notif))?);

                fn make_notification(event: $crate::Event<Self, Self::Key, Self::NotificationValue>) -> <Self::Storage as $crate::schema::GeneratedStorage>::Notification {
                    Notification::$name(event)
                }
                fn clone_value_for_notification(_value: &Self::Value) -> Self::NotificationValue {
                    $crate::notification_clone_value!(_value $(; notify($notif))?)
                }
            }

            impl $crate::schema::Indexable for $name {
                type Indexes<'store> = index::$name::Indexes<'store>;
                type IndexesWithOwner<'store> = index::$name::IndexesWithOwner<'store>;
                type TransactionIndexes<'guard, 'txn> = index::$name::TransactionIndexes<'guard, 'txn>;
                type RoTransactionIndexes<'guard, 'txn> = index::$name::RoTransactionIndexes<'guard, 'txn>;
            }

            $(
                impl $crate::schema::TableDesc for index::$name::$field where $field_ty: Clone {
                    const NAME: &'static str = stringify!($name by $field);
                    const OWNER: $crate::Owner = $owner;
                    type Key = $field_ty;
                    type Value = $key_ty;
                    type Storage = TableStorage;
                    type IndexStorage = ();

                    fn get_table(storage: &TableStorage) -> &$crate::storage::Table<Self, Self::IndexStorage> {
                        &storage.$name.indexes.$field
                    }
                    unsafe fn table_ptr(storage: *mut TableStorage) -> *mut $crate::storage::Table<Self, Self::IndexStorage> {
                        // SAFETY: `storage` is valid by the precondition of `table_ptr`; only a raw
                        // pointer is created.
                        unsafe { &raw mut (*storage).$name.indexes.$field }
                    }
                    fn make_txn_view<'a, 'b>(_txn: &'b mut <Self::Storage as $crate::schema::GeneratedStorage>::Transaction<'a>) -> &'b mut $crate::TableTransaction<'a, Self::Storage, Self> {
                        unreachable!()
                    }

                    $crate::value_eq!(Self::Value);
                }

                impl $crate::schema::IndexDesc for index::$name::$field {
                    type BaseTable = $name;

                    fn index(base: &$crate::storage::Table<$name, index::$name::Storage>) -> &$crate::storage::Table<Self, Self::IndexStorage> {
                        &base.indexes.$field
                    }

                    fn index_mut(base: &mut $crate::storage::Table<$name, index::$name::Storage>) -> &mut $crate::storage::Table<Self, Self::IndexStorage> {
                        &mut base.indexes.$field
                    }
                }
            )*
        )*)?

        /// Macro-generated storage for all data.
        #[derive(Default)]
        #[allow(non_snake_case)]
        pub struct TableStorage {
            $($($name: $crate::storage::Table<$name, index::$name::Storage>,)*)?
            $($($sname: $crate::storage::VersionedValue<Option<$svalue_ty>>,)*)?
        }

        /// Macro-generated notification type, there is a variant for each table and singleton with
        /// the appropriate types (wrapping event types in `ts_kv_store`). For notifications which
        /// don't include a value (either by default or by opting-out using `notify(None)`, the value
        /// type is `()`.
        #[derive(Clone)]
        #[allow(unused)]
        pub enum Notification {
            $($($name($crate::Event<$name, <$name as $crate::schema::TableDesc>::Key, <$name as $crate::schema::Notifiable>::NotificationValue>),)*)?
            $($($sname($crate::SingletonEvent<$sname, <$sname as $crate::schema::SingletonDesc>::NotificationValue>),)*)?
        }

        impl std::fmt::Debug for Notification {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $($(Notification::$name(..) => write!(f, stringify!($name)),)*)?
                    $($(Notification::$sname(..) => write!(f, stringify!($sname)),)*)?
                }
            }
        }

        impl $crate::schema::GeneratedStorage for TableStorage {
            type Notification = Notification;
            type Transaction<'a> = Transaction<'a>;
            type RoTransaction<'a> = RoTransaction<'a>;

            fn commit_txn(&mut self, _txn_id: $crate::transactions::TxnId, _notifications: &mut $crate::Notifications<Notification>, _subscriptions: &$crate::pub_sub::Subscriptions<Self::Notification>) -> $crate::Result<()> {
                $(
                    $(
                        self.$name.check_txn_consistency(_txn_id)?;
                        $(self.$name.indexes.$field.check_txn_consistency(_txn_id)?;)*
                    )*
                )?
                $(
                    $(
                        if _subscriptions.has_singleton_subscribers::<$sname>()
                            && let Some(event) = self.$sname.modified_in_txn(_txn_id)
                        {
                            _subscriptions.collect_singleton_events::<$sname>(_notifications, <$sname as $crate::schema::SingletonDesc>::notif_value(event));
                        }
                    )*
                )?
                $(
                    $(
                        let events = self.$name.commit_primary_table(_txn_id, _subscriptions.has_subscribers::<$name>());
                        if !events.is_empty() {
                            _subscriptions.collect_events::<$name>(_notifications, events);
                        }
                        $(self.$name.indexes.$field.commit_index(_txn_id);)*
                    )*
                )?

                Ok(())
            }

            fn gc_txn(&mut self, _txn_id: $crate::transactions::TxnId) {
                $(
                    $(
                        self.$name.gc_txn(_txn_id);
                        $(self.$name.indexes.$field.gc_txn(_txn_id);)*
                    )*
                )?
                $(
                    $(
                        self.$sname.gc_txn(_txn_id);
                    )*
                )?
            }

            fn make_txn<'a>(store_txn: $crate::Transaction<'a, Self>) -> Self::Transaction<'a> {
                Transaction::new(store_txn)
            }

            fn make_ro_txn<'a>(store_txn: $crate::RoTransaction<'a, Self>) -> Self::RoTransaction<'a> {
                RoTransaction::new(store_txn)
            }
        }

        pub mod index {
            $($(
                #[allow(non_snake_case)]
                pub mod $name {
                    $(
                        #[allow(non_camel_case_types)]
                        pub struct $field;
                    )*

                    /// Storage for the table's indexes.
                    #[derive(Default)]
                    pub struct Storage {
                        $(
                            #[allow(dead_code)]
                            pub $field: $crate::storage::Table<$field, ()>,
                        )*
                    }

                    /// Access to the table's indexes, with a field for each index.
                    ///
                    /// Returned by `Table::indexes`.
                    pub struct Indexes<'store> {
                        #[doc(hidden)]
                        pub(in super::super) _store: core::marker::PhantomData<&'store ()>,
                        $(
                            #[allow(dead_code)]
                            pub $field: $crate::Index<'store, $field>,
                        )*
                    }

                    /// Access to the table's indexes with a fixed owner, with a field for each index.
                    ///
                    /// Returned by `TableWithOwner::indexes`.
                    pub struct IndexesWithOwner<'store> {
                        #[doc(hidden)]
                        pub(in super::super) _store: core::marker::PhantomData<&'store ()>,
                        $(
                            #[allow(dead_code)]
                            pub $field: $crate::IndexWithOwner<'store, $field>,
                        )*
                    }

                    /// Access to the table's indexes within a transaction, with a field for each index.
                    ///
                    /// Returned by `TableTransaction::indexes`.
                    pub struct TransactionIndexes<'guard, 'txn> {
                        #[doc(hidden)]
                        pub(in super::super) _txn: core::marker::PhantomData<(&'txn mut (), &'guard ())>,
                        $(
                            #[allow(dead_code)]
                            pub $field: $crate::IndexTransaction<'guard, 'txn, $field>,
                        )*
                    }

                    /// Access to the table's indexes within a read-only transaction, with a field
                    /// for each index.
                    ///
                    /// Returned by `RoTableTransaction::indexes`.
                    pub struct RoTransactionIndexes<'guard, 'txn> {
                        #[doc(hidden)]
                        pub(in super::super) _txn: core::marker::PhantomData<(&'txn (), &'guard ())>,
                        $(
                            #[allow(dead_code)]
                            pub $field: $crate::RoIndexTransaction<'guard, 'txn, $field>,
                        )*
                    }
                }
            )*)?
        }

        $($(
            impl<'store> From<&'store $crate::KvStore<TableStorage>> for index::$name::Indexes<'store> {
                fn from(_store: &'store $crate::KvStore<TableStorage>) -> Self {
                    index::$name::Indexes {
                        _store: core::marker::PhantomData,
                        $($field: $crate::Index::new(_store),)*
                    }
                }
            }

            impl<'store> From<(&'store $crate::KvStore<TableStorage>, $crate::Owner)> for index::$name::IndexesWithOwner<'store> {
                fn from((_store, _owner): (&'store $crate::KvStore<TableStorage>, $crate::Owner)) -> Self {
                    index::$name::IndexesWithOwner {
                        _store: core::marker::PhantomData,
                        $($field: $crate::IndexWithOwner::new(_store, _owner),)*
                    }
                }
            }

            impl<'guard, 'txn> From<*mut $crate::Transaction<'guard, TableStorage>> for index::$name::TransactionIndexes<'guard, 'txn> {
                fn from(_txn: *mut $crate::Transaction<'guard, TableStorage>) -> Self {
                    index::$name::TransactionIndexes {
                        _txn: core::marker::PhantomData,
                        $($field: unsafe { $crate::IndexTransaction::new(_txn) },)*
                    }
                }
            }

            impl<'guard, 'txn> From<*const $crate::RoTransaction<'guard, TableStorage>> for index::$name::RoTransactionIndexes<'guard, 'txn> {
                fn from(_txn: *const $crate::RoTransaction<'guard, TableStorage>) -> Self {
                    index::$name::RoTransactionIndexes {
                        _txn: core::marker::PhantomData,
                        $($field: unsafe { $crate::RoIndexTransaction::new(_txn) },)*
                    }
                }
            }

            impl index::$name::Storage {
                $(
                    fn $field(val: &$value_ty) -> impl IntoIterator<Item = $field_ty> {
                        ($crate::get_index_fn!($value_ty, $field $(, $get_idx)?))(val)
                    }
                )*
            }

            impl $crate::schema::IndexStorage<$key_ty, $value_ty> for index::$name::Storage {
                fn clear(&mut self, _txn_id: $crate::transactions::TxnId, _max_committed_id: $crate::transactions::TxnId) {
                    $(
                        self.$field.clear(_txn_id, _max_committed_id);
                    )*
                }

                $crate::on_insert!($name, $key_ty, $value_ty, $(index($field: $field_ty $(; $unique)?),)*);

                fn on_remove(&mut self, _value: &$value_ty, _txn_id: $crate::transactions::TxnId, _max_committed_id: $crate::transactions::TxnId) {
                    $({
                        for value in index::$name::Storage::$field(_value) {
                            self.$field.remove(&value, _txn_id, _max_committed_id);
                        }
                    })*
                }
            }
        )*)?

        /// A key-value store.
        ///
        /// See [`$crate::KvStore`] (which this type implicitly derefences to) for full docs.
        #[allow(non_snake_case)]
        pub struct KvStore {
            store: Box<$crate::KvStore<TableStorage>>,

            $($(#[allow(dead_code)] pub $name: $crate::Table<TableStorage, $name>,)*)?
            $($(#[allow(dead_code)] pub $sname: $crate::Singleton<TableStorage, $sname>,)*)?
        }

        impl KvStore {
            /// Create a new, empty KV store as described by the schema macros.
            ///
            /// The store has a no-op notifier, so subscribers are never notified of changes.
            #[allow(dead_code, clippy::new_without_default)]
            pub fn new() -> Self {
                // The store only holds a weak reference to its notifier, so downgrading a
                // throwaway no-op notifier leaves the store with a dead weak reference. Upgrading it
                // always fails, which means no notifications are ever sent.
                Self::from_notifier(std::sync::Arc::downgrade(&$crate::NoOpNotifier::new()))
            }

            /// Create a new, empty KV store as described by the schema macros, which sends
            /// notifications of changes to `notifier`.
            ///
            /// The store keeps only a weak reference to `notifier`, so the caller is responsible for
            /// keeping the notifier alive (e.g. via the subscribers it hands out). Once the last
            /// strong reference is dropped the store stops sending notifications.
            pub fn from_notifier(notifier: std::sync::Weak<dyn $crate::Notifier<Notification = <TableStorage as $crate::schema::GeneratedStorage>::Notification>>) -> Self {
                let store = Box::new($crate::KvStore::new_with_storage(std::sync::RwLock::new($crate::storage::Storage::new(notifier))));
                let raw = store.as_ref() as *const $crate::KvStore<_>;
                KvStore {
                    $($($name: unsafe { $crate::Table::new(raw) },)*)?
                    $($($sname: unsafe { $crate::Singleton::new(raw) },)*)?
                    store,
                }
            }

            /// A convenience for operating on the store with a specified owner.
            #[allow(dead_code)]
            pub fn with_owner(&self, owner: $crate::Owner) -> KvStoreWithOwner<'_> {
                let store = self.store.as_ref();
                KvStoreWithOwner {
                    $($($name: $crate::TableWithOwner::new(store, owner),)*)?
                    $($($sname: $crate::SingletonWithOwner::new(store, owner),)*)?
                    store,
                    owner,
                }
            }
        }

        impl From<std::sync::Weak<dyn $crate::Notifier<Notification = <TableStorage as $crate::schema::GeneratedStorage>::Notification>>> for KvStore {
            fn from(notifier: std::sync::Weak<dyn $crate::Notifier<Notification = <TableStorage as $crate::schema::GeneratedStorage>::Notification>>) -> KvStore {
                KvStore::from_notifier(notifier)
            }
        }

        // Blocks moving the per-table/singleton fields out of the store: they hold pointers
        // back into it, so they must not outlive it.
        impl Drop for KvStore {
            fn drop(&mut self) {}
        }

        impl std::ops::Deref for KvStore {
            type Target = $crate::KvStore<TableStorage>;

            fn deref(&self) -> &Self::Target {
                self.store.as_ref()
            }
        }

        impl $crate::GeneratedStore<TableStorage> for KvStore {}

        /// A key-value store with a fixed owner.
        ///
        /// Created by `KvStore::with_owner`. Operations on its fields do not take an owner, they
        /// use the owner supplied to `with_owner`.
        #[allow(non_snake_case)]
        pub struct KvStoreWithOwner<'a> {
            store: &'a $crate::KvStore<TableStorage>,
            owner: $crate::Owner,

            $($(#[allow(dead_code)] pub $name: $crate::TableWithOwner<'a, TableStorage, $name>,)*)?
            $($(#[allow(dead_code)] pub $sname: $crate::SingletonWithOwner<'a, TableStorage, $sname>,)*)?
        }

        // Blocks moving the per-table/singleton fields out of the store: they hold pointers back
        // into the store this borrows, and the borrow is what keeps them valid.
        impl<'a> Drop for KvStoreWithOwner<'a> {
            fn drop(&mut self) {}
        }

        #[allow(dead_code)]
        impl<'a> KvStoreWithOwner<'a> {
            /// Start a transaction.
            ///
            /// Blocks until the store's global lock is available for write access.
            pub fn begin_transaction(&self) -> Transaction<'a> {
                self.store.begin_transaction(self.owner)
            }

            /// Start a transaction.
            ///
            /// Returns `None` if the store's global lock is unavailable for write access.
            pub fn try_begin_transaction(&self) -> Option<Transaction<'a>> {
                self.store.try_begin_transaction(self.owner)
            }

            /// Start a read-only transaction (i.e., only supports non-mutating access to the store,
            /// but all reads are guaranteed to be atomic).
            ///
            /// Blocks until the store's global lock is available for read access.
            pub fn begin_ro_transaction(&self) -> RoTransaction<'a> {
                self.store.begin_ro_transaction(self.owner)
            }

            /// Start a read-only transaction (i.e., only supports non-mutating access to the store,
            /// but all reads are guaranteed to be atomic).
            ///
            /// Returns `None` if the store's global lock is unavailable for read access.
            pub fn try_begin_ro_transaction(&self) -> Option<RoTransaction<'a>> {
                self.store.try_begin_ro_transaction(self.owner)
            }

            /// Register a new subscriber (with the owner supplied to `with_owner`) to the store.
            ///
            /// Does not create any subscriptions.
            pub fn register_subscriber(&self) -> $crate::Subscriber {
                self.store.register_subscriber(self.owner)
            }

            /// Remove a subscriber from the store, along with all of its subscriptions.
            ///
            /// The subscriber receives no further notifications and cannot subscribe again
            /// (subscribing with a removed subscriber gives [`$crate::Error::UnknownSubscriber`]).
            /// Unsubscribing any of its subscriptions is harmless but pointless.
            ///
            /// Does nothing if the subscriber is unknown (e.g., because it has already been
            /// removed).
            pub fn remove_subscriber(&self, subscriber: $crate::Subscriber) {
                self.store.remove_subscriber(subscriber)
            }

            /// Subscribe to the whole store.
            pub fn subscribe_global(
                &self,
                subscriber: $crate::Subscriber,
            ) -> $crate::Result<$crate::Subscription> {
                self.store.subscribe_global(subscriber)
            }

            /// Remove any subscriptions to the whole store.
            pub fn unsubscribe_global(&self, subscription: $crate::Subscription) {
                self.store.unsubscribe_global(subscription);
            }
        }

        #[allow(non_snake_case)]
        pub struct Transaction<'a> {
            // `None` only once the transaction has been committed; dropping a `Some` rolls the
            // transaction back and releases the store's lock.
            store_txn: Option<Box<$crate::Transaction<'a, TableStorage>>>,

            $($(#[allow(dead_code)] pub $name: $crate::TableTransaction<'a, TableStorage, $name>,)*)?
            $($(#[allow(dead_code)] pub $sname: $crate::SingletonTransaction<'a, TableStorage, $sname>,)*)?
        }

        impl<'a> Transaction<'a> {
            fn new(store_txn: $crate::Transaction<'a, TableStorage>) -> Self {
                let mut store_txn = Box::new(store_txn);
                let raw = store_txn.as_mut() as *mut $crate::Transaction<'a, TableStorage>;

                Transaction {
                    $($($name: unsafe { $crate::TableTransaction::new(raw) },)*)?
                    $($($sname: unsafe { $crate::SingletonTransaction::new(raw) },)*)?

                    store_txn: Some(store_txn),
                }
            }

            pub fn commit(mut self) -> $crate::Result<()> {
                // `Self` implements `Drop` (so that the per-table fields cannot be moved out of a
                // transaction), which means `store_txn` cannot be moved out of `self` either; take
                // it instead. Dropping `self` afterwards then has nothing left to roll back.
                let store_txn = self.store_txn.take().expect("a transaction is committed once");
                store_txn.commit()
            }

            pub fn rollback(self) {
                // Dropping `self` causes the rollback.
            }
        }

        // Blocks moving the per-table/singleton fields out of the transaction: they hold pointers
        // back into it, so they must not outlive it. Rolling back an uncommitted transaction is
        // left to the drop glue for `store_txn`.
        impl<'a> Drop for Transaction<'a> {
            fn drop(&mut self) {}
        }

        impl<'a> $crate::transactions::SchemaTransaction for Transaction<'a> {
            fn commit(self) -> $crate::Result<()> {
                self.commit()
            }

            fn rollback(self) {
                self.rollback()
            }

        }

        #[allow(non_snake_case)]
        pub struct RoTransaction<'a> {
            _store_txn: Box<$crate::RoTransaction<'a, TableStorage>>,

            $($(#[allow(dead_code)] pub $name: $crate::RoTableTransaction<'a, TableStorage, $name>,)*)?
            $($(#[allow(dead_code)] pub $sname: $crate::RoSingletonTransaction<'a, TableStorage, $sname>,)*)?
        }

        impl<'a> RoTransaction<'a> {
            fn new(store_txn: $crate::RoTransaction<'a, TableStorage>) -> Self {
                let _store_txn = Box::new(store_txn);
                let raw = _store_txn.as_ref() as *const $crate::RoTransaction<'a, TableStorage>;

                RoTransaction {
                    $($($name: unsafe { $crate::RoTableTransaction::new(raw) },)*)?
                    $($($sname: unsafe { $crate::RoSingletonTransaction::new(raw) },)*)?

                    _store_txn,
                }
            }

            pub fn commit(self) -> $crate::Result<()> {
                Ok(())
            }

            pub fn rollback(self) {
                // Dropping `self` causes the rollback.
            }
        }

        // Blocks moving the per-table/singleton fields out of the transaction: they hold pointers
        // back into it, so they must not outlive it.
        impl<'a> Drop for RoTransaction<'a> {
            fn drop(&mut self) {}
        }

        impl<'a> $crate::transactions::SchemaTransaction for RoTransaction<'a> {
            fn commit(self) -> $crate::Result<()> {
                self.commit()
            }

            fn rollback(self) {
                self.rollback()
            }

        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! get_index_fn {
    ($value_ty:ty, $field:ident) => {
        |value: &$value_ty| [value.$field.clone()]
    };
    ($value_ty:ty, $field:ident, $get_idx:expr) => {
        $get_idx
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! on_insert {
    ($name: ident, $key_ty: ty, $value_ty: ty, $(index($field: ident: $field_ty: ty $(; $unique:ident)?),)*) => {
        fn on_insert<Q>(&mut self, _key: &Q, _value: &$value_ty, _txn_id: $crate::transactions::TxnId, _max_committed_id: $crate::transactions::TxnId)
        where
            $key_ty: std::borrow::Borrow<Q>,
            Q: ?Sized + std::hash::Hash + Eq + std::borrow::ToOwned<Owned = $key_ty>
        {
            $(
                $crate::on_insert_each!($name, $field: $field_ty; (self, _key, _value, _txn_id, _max_committed_id) $(; $unique)?);
            )*
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! on_insert_each {
    (
        $name:ident,
        $field:ident :
        $field_ty:ty;
        ($self:ident, $key:ident, $value:ident, $txn_id:ident, $max_committed_id:ident); assert_unique
    ) => {
        for index_key in index::$name::Storage::$field($value) {
            let unique = $self.$field.get::<$field_ty>(&index_key, $txn_id).is_none();
            assert!(
                unique,
                "Index key is non-unique for index `{}` of table `{}`",
                stringify!($field),
                stringify!($name),
            );
            $self
                .$field
                .insert(index_key, $key.to_owned(), $txn_id, $max_committed_id);
        }
    };
    (
        $name:ident,
        $field:ident :
        $field_ty:ty;
        ($self:ident, $key:ident, $value:ident, $txn_id:ident, $max_committed_id:ident)
    ) => {
        for index_key in index::$name::Storage::$field($value) {
            let unique = $self.$field.get::<$field_ty>(&index_key, $txn_id).is_none();
            if unique {
                $self
                    .$field
                    .insert(index_key, $key.to_owned(), $txn_id, $max_committed_id);
            } else {
                $self.$field.set_poisoned($txn_id);
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! value_eq {
    ($t:ty) => {
        fn value_eq(a: &$t, b: &$t) -> bool {
            // Use the 'autoref specialization' trick (https://github.com/dtolnay/case-studies/tree/master/autoref-specialization)
            // to compare values if possible and panic if not. Panicking is safe here because
            // this method is only called for possibly mutated values, and values can only be
            // mutated if they impl `PartialEq`.
            #[allow(dead_code)]
            trait HasEq {
                fn veq(&self, other: &Self) -> bool;
            }
            #[allow(dead_code)]
            trait MaybeEq {
                fn veq(&self, other: Self) -> bool;
            }
            impl<T: core::cmp::PartialEq> HasEq for T {
                fn veq(&self, other: &Self) -> bool {
                    self == other
                }
            }
            impl<T> MaybeEq for &T {
                fn veq(&self, _other: Self) -> bool {
                    unreachable!();
                }
            }

            a.veq(b)
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! notification_value_type {
    ($value_ty:ty; notify(Clone)) => {
        $value_ty
    };
    ($value_ty:ty $(; notify(None))?) => {
        ()
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! notification_clone_value {
    ($value:ident; notify(Clone)) => {
        $value.clone()
    };
    ($value:ident $(; notify(None))?) => {
        ()
    };
}
#[doc(hidden)]
#[macro_export]
macro_rules! notification_value {
    ($value:ident; notify(Clone)) => {
        $value
    };
    ($value:ident $(; notify(None))?) => {
        if $value.is_some() { &Some(()) } else { &None }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! get_cloned_notification_value {
    ($name:ident, $storage:ident, $txn_id:ident; notify(Clone)) => {
        $storage.$name.get($txn_id)?.as_ref().map(|v| v.clone())
    };
    ($name:ident, $storage:ident, $txn_id:ident $(; notify(None))?) => {
        $storage.$name.get($txn_id)?.as_ref().map(|_| ())
    };
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    #[test]
    fn single() {
        store!(
            kvs: {
                Foo(Box<u64>; "owner"; notify(Clone)),
                Bar(Arc<u64>; "owner"; notify(None)),
                Baz(&'static u64; "owner"),
                Qux(u64; "owner"),
            }
        );

        let store = KvStore::new();
        store.Foo.insert("owner", Box::new(42));
        assert_eq!(store.Foo.get("owner").unwrap(), Box::new(42));
    }

    #[test]
    fn table() {
        store!(tables: { Foo(&'static str => String; "owner"; notify(Clone)), Bar(u32 => Vec<String>; "owner")});

        let store = KvStore::new();

        store.Foo.insert("owner", "hello", "world".to_owned());
        assert_eq!(store.Foo.get("owner", "hello").unwrap(), "world");

        store
            .Bar
            .insert("owner", 5, vec!["boo".to_owned(), "bang".to_owned()]);
        assert_eq!(
            store.Bar.get("owner", &5).unwrap(),
            vec!["boo".to_owned(), "bang".to_owned()]
        );
    }

    #[test]
    fn send_and_sync() {
        fn require_send<T: Send>(_t: T) {}
        fn require_sync<T: Sync>(_t: T) {}

        store!(tables: { Foo(&'static str => String; "owner"; notify(Clone)), Bar(u32 => Vec<String>; "owner")});

        require_send(KvStore::new());
        require_sync(KvStore::new());
    }

    #[test]
    fn table_with_indexes() {
        #[derive(Clone, Debug)]
        pub struct BarT {
            a: String,
        }
        store!(
            tables: {
                Foo(&'static str => String; "owner"; index(len: usize = |v: &String| [v.len()])),
                Bar(u32 => BarT; "owner"; index(a: String; assert_unique)),
            }
        );

        let store = KvStore::new();
        store.Bar.insert(
            "owner",
            5,
            BarT {
                a: "hello".to_owned(),
            },
        );
        let value = store.Bar.indexes().a.get("owner", "hello").unwrap();
        assert_eq!(value.1.a, "hello");

        store.Foo.insert("owner", "foo", "hello".to_owned());
        let value = store.Foo.indexes().len.get("owner", &5).unwrap();
        assert_eq!(value, ("foo", "hello".to_owned()));
    }
}
