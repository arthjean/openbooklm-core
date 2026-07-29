//! Public core account entity (US-011).
//!
//! The owner of notebooks, sources, notes and memories. It carries an
//! identifier and two timestamps and nothing else: no email, no provider
//! subject, no plan. Who the account *is* belongs to an identity adapter; what
//! the account is *allowed* to do belongs to an entitlement policy.
//!
//! `accounts.id` is the same UUID the legacy `users.id` held, so ownership
//! foreign keys are unchanged by the split.

use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "accounts")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,
    pub created_at: DateTimeWithTimeZone,
    pub updated_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(has_one = "super::account_settings::Entity")]
    Settings,
}

super::impl_related!(super::account_settings::Entity, Relation::Settings);

impl ActiveModelBehavior for ActiveModel {}
