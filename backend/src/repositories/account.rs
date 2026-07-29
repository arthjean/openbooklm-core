//! Public core account and settings repositories (US-011).
//!
//! `accounts` is the ownership root the core knows about, and
//! `account_settings` holds the preferences it implements. Neither touches an
//! email address, an identity subject or a plan: those belong to the private
//! identity and entitlement adapters.

use async_trait::async_trait;
use chrono::Utc;
use sea_orm::{
    ActiveModelTrait, ConnectionTrait, DatabaseBackend, DatabaseConnection, EntityTrait, Set,
    Statement,
};
use uuid::Uuid;

use crate::entities::{Account, AccountSettings, account, account_settings};
use crate::error::{AppError, SettingsError};

use super::traits::{AccountRepository, AccountSettingsRepository};

/// Defaults for a brand-new account. Mistral is the cheapest provider every
/// edition can reach, so a self-hosted install works with one API key.
const DEFAULT_PROVIDER: &str = "mistral";
const DEFAULT_MODEL: &str = "mistral-small-latest";

#[derive(Clone)]
pub struct SeaOrmAccountRepository {
    db: DatabaseConnection,
}

impl SeaOrmAccountRepository {
    #[must_use]
    pub fn new(db: &DatabaseConnection) -> Self {
        Self { db: db.clone() }
    }
}

#[async_trait]
impl AccountRepository for SeaOrmAccountRepository {
    #[tracing::instrument(skip(self), fields(%account_id))]
    async fn ensure_exists(&self, account_id: Uuid) -> Result<account::Model, AppError> {
        if let Some(existing) = Account::find_by_id(account_id).one(&self.db).await? {
            return Ok(existing);
        }

        let now = Utc::now();
        self.db
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                r"INSERT INTO accounts (id, created_at, updated_at)
                  VALUES ($1, $2, $2)
                  ON CONFLICT (id) DO NOTHING",
                [account_id.into(), now.into()],
            ))
            .await?;

        Account::find_by_id(account_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| AppError::Internal("Account missing after upsert".into()))
    }

    #[tracing::instrument(skip(self), fields(%account_id))]
    async fn find(&self, account_id: Uuid) -> Result<Option<account::Model>, AppError> {
        Ok(Account::find_by_id(account_id).one(&self.db).await?)
    }
}

#[derive(Clone)]
pub struct SeaOrmAccountSettingsRepository {
    db: DatabaseConnection,
}

impl SeaOrmAccountSettingsRepository {
    #[must_use]
    pub fn new(db: &DatabaseConnection) -> Self {
        Self { db: db.clone() }
    }
}

#[async_trait]
impl AccountSettingsRepository for SeaOrmAccountSettingsRepository {
    #[tracing::instrument(skip(self), fields(%account_id))]
    async fn get_or_create(&self, account_id: Uuid) -> Result<account_settings::Model, AppError> {
        if let Some(existing) = AccountSettings::find_by_id(account_id)
            .one(&self.db)
            .await?
        {
            return Ok(existing);
        }

        // Race-safe: two concurrent first requests both insert, one wins, both
        // read the winner's row.
        let now = Utc::now();
        self.db
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                r"INSERT INTO account_settings (account_id, default_provider, default_model, updated_at)
                  VALUES ($1, $2, $3, $4)
                  ON CONFLICT (account_id) DO NOTHING",
                [
                    account_id.into(),
                    DEFAULT_PROVIDER.into(),
                    DEFAULT_MODEL.into(),
                    now.into(),
                ],
            ))
            .await
            .map_err(|e| {
                tracing::error!(error = %e, %account_id, "Failed to upsert account settings");
                SettingsError::Database(e)
            })?;

        AccountSettings::find_by_id(account_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| {
                tracing::error!(%account_id, "Account settings missing after upsert");
                SettingsError::NotFound {
                    user_id: account_id.to_string(),
                }
                .into()
            })
    }

    #[tracing::instrument(skip(self), fields(%account_id))]
    async fn update_defaults(
        &self,
        account_id: Uuid,
        default_provider: Option<String>,
        default_model: Option<String>,
    ) -> Result<account_settings::Model, AppError> {
        let existing = self.get_or_create(account_id).await?;
        let mut active: account_settings::ActiveModel = existing.into();

        if let Some(provider) = default_provider {
            active.default_provider = Set(provider);
        }
        if let Some(model) = default_model {
            active.default_model = Set(model);
        }
        active.updated_at = Set(Utc::now().into());

        active.update(&self.db).await.map_err(|e| {
            tracing::error!(error = %e, %account_id, "Account settings update failed");
            SettingsError::Database(e).into()
        })
    }
}
