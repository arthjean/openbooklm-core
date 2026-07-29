//! Notebook entity - the primary organizational unit.
//!
//! A notebook groups related sources and chat conversations.
//! Users can have multiple notebooks (limited by subscription plan).
//!
//! | Plan  | Max Notebooks |
//! |-------|---------------|
//! | Free  | 3             |
//! | Pro   | 10            |
//! | Pro+  | Unlimited     |
//! | Max   | Unlimited     |

use sea_orm::entity::prelude::*;

/// Notebook containing sources and chat history.
#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "notebooks")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    /// Owner of this notebook
    pub user_id: Uuid,

    /// Display name
    pub title: String,

    /// Optional description/notes
    pub description: Option<String>,

    /// Whether memory extraction is enabled for this notebook
    pub memory_enabled: bool,

    /// Whether this is the auto-provisioned demo notebook
    pub is_demo: bool,

    /// Suggested questions for this notebook (JSONB array of strings)
    #[sea_orm(column_type = "JsonBinary")]
    pub suggested_questions: serde_json::Value,

    pub created_at: DateTimeWithTimeZone,
    pub updated_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::account::Entity",
        from = "Column::UserId",
        to = "super::account::Column::Id"
    )]
    Account,
    #[sea_orm(has_many = "super::source::Entity")]
    Sources,
    #[sea_orm(has_many = "super::chat_message::Entity")]
    ChatMessages,
    #[sea_orm(has_many = "super::notebook_memory::Entity")]
    NotebookMemories,
}

// SeaORM Related implementations for eager loading
super::impl_related!(super::account::Entity, Relation::Account);
super::impl_related!(super::source::Entity, Relation::Sources);
super::impl_related!(super::chat_message::Entity, Relation::ChatMessages);
super::impl_related!(super::notebook_memory::Entity, Relation::NotebookMemories);

impl ActiveModelBehavior for ActiveModel {}
