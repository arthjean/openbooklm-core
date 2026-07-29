/**
 * `@openbooklm/sdk` — the typed client for the OpenbookLM core API.
 *
 * Everything exported here is generated from, or aliased to, the Rust
 * definitions in the public core. Nothing commercial crosses this boundary:
 * billing plans, price identifiers, identity-provider claims, analytics events
 * and proprietary UI types stay in the private SaaS repository.
 */

export { OpenbookLMClient, type ClientOptions } from "./client.js";

export {
  ApiError,
  type ChatHistory,
  type ChatMessage,
  type Chunk,
  type ChunkList,
  type Citation,
  type CreateNoteRequest,
  type CreateNotebookRequest,
  type CreateSourceRequest,
  type DetailedHealth,
  type Health,
  type Memory,
  type MemoryList,
  type Metrics,
  type Note,
  type NoteList,
  type Notebook,
  type NotebookList,
  type ProblemDetails,
  type SendMessageRequest,
  type Settings,
  type Source,
  type SourceList,
  type Suggestions,
  type TeachingModeInfo,
  type TeachingModes,
  type UpdateFeedbackRequest,
  type UpdateMemoryRequest,
  type UpdateNoteRequest,
  type UpdateNotebookRequest,
  type UpdateSettingsRequest,
  type YouTubeTitle,
  type components,
  type operations,
  type paths,
} from "./types.js";

export {
  CHAT_TERMINAL_EVENTS,
  isChatTerminal,
  parseChatEvent,
  parseSourceEvent,
  readEventStream,
  type ChatChunk,
  type ChatCitationRef,
  type ChatCitations,
  type ChatDone,
  type ChatError,
  type ChatEvent,
  type ChatEventName,
  type ChatEventOf,
  type ChatFollowUpSuggestions,
  type ChatMetrics,
  type ChatShutdown,
  type ChatSystem,
  type ChatThinking,
  type ChatWarning,
  type EmbeddingProgress,
  type RawEvent,
  type SourceEvent,
  type SourceEventName,
  type SourceEventOf,
  type SourceStatusData,
} from "./events.js";

export {
  CORE_CATALOG,
  DEFAULT_TEACHING_MODE,
  EVENT_PROTOCOL_VERSION,
  PROVIDERS,
  SOURCE_STATUSES,
  SOURCE_TYPES,
  TEACHING_MODES,
  VALIDATION_LIMITS,
  type ProviderName,
  type SourceStatus,
  type SourceType,
  type TeachingMode,
} from "./generated/catalog.js";
