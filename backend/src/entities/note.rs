//! User notes entity for saving insights from chat conversations.
//!
//! Notes allow users to save and organize key information extracted
//! from AI responses. Can be created manually or from a chat message.

use sea_orm::entity::prelude::*;

/// User-created note within a notebook.
#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "notes")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    pub notebook_id: Uuid,

    /// Note title (user-provided or auto-generated)
    pub title: String,

    /// Markdown content
    pub content: String,

    /// Links to chat_messages.id if created from "Save as note"
    pub original_message_id: Option<Uuid>,

    pub created_at: DateTimeWithTimeZone,
    pub updated_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::notebook::Entity",
        from = "Column::NotebookId",
        to = "super::notebook::Column::Id"
    )]
    Notebook,
}

super::impl_related!(super::notebook::Entity, Relation::Notebook);

impl ActiveModelBehavior for ActiveModel {}
