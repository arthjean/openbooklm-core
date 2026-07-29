//! Public core account settings (US-011).
//!
//! Only preferences the core actually implements. Onboarding progress and
//! campaign state used to share this row; they now live in
//! `saas_account_settings`, which the public edition never creates.

use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "account_settings")]
pub struct Model {
    /// One row per account.
    #[sea_orm(primary_key, auto_increment = false)]
    pub account_id: Uuid,

    /// Default LLM provider: "anthropic" | "openai" | "mistral".
    pub default_provider: String,

    /// Default model ID (e.g. "claude-sonnet-4-6-20260220").
    pub default_model: String,

    pub updated_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::account::Entity",
        from = "Column::AccountId",
        to = "super::account::Column::Id"
    )]
    Account,
}

super::impl_related!(super::account::Entity, Relation::Account);

impl ActiveModelBehavior for ActiveModel {}
