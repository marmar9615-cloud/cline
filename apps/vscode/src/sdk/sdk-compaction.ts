// Replaces classic manual-condense handling from src/core/task (see origin/main)
//
// Manual context compaction for the VSCode SDK adapter. This mirrors the CLI's
// apps/cli/src/runtime/interactive/compaction.ts (`compactInteractiveMessages`):
// it builds a manual-mode compaction `prepareTurn` via the SDK's
// `createContextCompactionPrepareTurn` and runs it against the session's
// working context (the existing sidecar projected over the canonical
// transcript), returning the new compacted working-context sidecar state.
//
// The VSCode coordinator persists that sidecar without replacing the canonical
// transcript, so the active session and later resumes use compacted working
// context while saved messages remain intact.

import {
	type CoreSessionConfig,
	createContextCompactionPrepareTurn,
	createSessionCompactionState,
	projectSessionCompactionState,
	type SessionCompactionState,
} from "@cline/core"
import type { Message as SdkMessage, ModelInfo as SdkModelInfo } from "@cline/llms"
import { Logger } from "@/shared/services/Logger"

// When the active model does not declare a context window, fall back to a
// conservative input budget so manual compaction still has a target to shrink
// toward. Matches the CLI's FALLBACK_MANUAL_COMPACTION_MAX_INPUT_TOKENS.
const FALLBACK_MANUAL_COMPACTION_MAX_INPUT_TOKENS = 64_000

export interface CompactSessionMessagesInput {
	/** Provider/model/compaction config for the active session. */
	config: Pick<
		CoreSessionConfig,
		"providerConfig" | "providerId" | "modelId" | "knownModels" | "compaction" | "logger" | "telemetry"
	>
	/** The active session id (used for telemetry keying). */
	sessionId: string
	/** The canonical conversation transcript (SDK message shape). */
	messages: SdkMessage[]
	/**
	 * The session's existing compacted working-context sidecar, if any. When
	 * it projects cleanly over `messages`, compaction runs on that projection
	 * — the transcript the model actually receives — instead of re-compacting
	 * the full canonical history. Canonical can outgrow the model's context
	 * window by millions of tokens on long sessions, and compacting it from
	 * scratch produces an over-window result that permanently wedges the
	 * session (cline/cline#12996).
	 */
	compactionState?: SessionCompactionState
	/**
	 * Receives the SDK's compaction status notices (started/completed/skipped
	 * with token + message counters) so the caller can drive progress UI.
	 */
	emitStatusNotice?: (message: string, metadata?: Record<string, unknown>) => void
}

export interface CompactSessionMessagesResult {
	compacted: boolean
	messages: SdkMessage[]
	compactionState?: SessionCompactionState
}

/**
 * Run a manual context compaction over the supplied messages.
 *
 * Returns `{ compacted: false }` (with the original messages) when there is
 * nothing to compact or the configured strategy declines to compact.
 */
export async function compactSessionMessages(input: CompactSessionMessagesInput): Promise<CompactSessionMessagesResult> {
	if (input.messages.length === 0) {
		return { compacted: false, messages: input.messages }
	}

	const modelInfo: SdkModelInfo | undefined = input.config.knownModels?.[input.config.modelId]
	const compactionModelInfo: SdkModelInfo = modelInfo
		? {
				...modelInfo,
				id: modelInfo.id ?? input.config.modelId,
			}
		: {
				id: input.config.modelId,
				maxInputTokens: FALLBACK_MANUAL_COMPACTION_MAX_INPUT_TOKENS,
			}

	const compact = createContextCompactionPrepareTurn(
		{
			providerConfig: input.config.providerConfig,
			providerId: input.config.providerId,
			modelId: input.config.modelId,
			// Force-enable compaction for this manual request even when
			// auto-condense is off — the user explicitly asked for it.
			compaction: {
				...input.config.compaction,
				enabled: true,
			},
			logger: input.config.logger,
			// Forward telemetry + sessionId so manual compactions emit
			// `task.compaction_executed` / `task.compaction_skipped` events,
			// matching the CLI and auto-compaction.
			telemetry: input.config.telemetry,
			sessionId: input.sessionId,
		},
		{ mode: "manual" },
	)
	if (!compact) {
		Logger.warn("[SdkCompaction] Compaction prepareTurn unavailable; skipping manual compaction")
		return { compacted: false, messages: input.messages }
	}

	// Compact the working context — the same transcript every turn sends to
	// the model — not the raw canonical history. Falls back to canonical when
	// there is no sidecar or it no longer projects (e.g. rewritten history).
	const projectedMessages = input.compactionState
		? projectSessionCompactionState(input.compactionState, input.messages)
		: undefined
	const workingMessages = projectedMessages ?? input.messages

	const result = await compact({
		agentId: "cline-vscode",
		conversationId: input.sessionId,
		parentAgentId: null,
		iteration: 0,
		messages: workingMessages,
		apiMessages: workingMessages,
		abortSignal: new AbortController().signal,
		systemPrompt: "",
		tools: [],
		model: {
			id: input.config.modelId,
			provider: input.config.providerId,
			info: compactionModelInfo,
		},
		emitStatusNotice: input.emitStatusNotice,
	})
	if (!result) {
		return { compacted: false, messages: workingMessages }
	}
	// The new sidecar stays keyed to the canonical transcript (same as the
	// SDK's own re-compaction flow), and a system prompt carried by the prior
	// sidecar survives unless this compaction rewrote it.
	const systemPrompt = result.systemPrompt ?? (projectedMessages ? input.compactionState?.system_prompt : undefined)
	return {
		compacted: true,
		messages: result.messages,
		compactionState: createSessionCompactionState({
			sourceMessages: input.messages,
			compactedMessages: result.messages,
			conversationId: input.sessionId,
			systemPrompt,
		}),
	}
}
