import {
	createContextCompactionPrepareTurn,
	createSessionCompactionState,
	type ProviderConfig,
	type ProviderSettings,
	type ProviderSettingsManager,
	projectSessionCompactionState,
	type ReasoningSettings,
	type SessionCompactionState,
	toProviderConfig,
} from "@cline/core";
import type { Message } from "@cline/shared";
import type { Config } from "../../utils/types";

const FALLBACK_MANUAL_COMPACTION_MAX_INPUT_TOKENS = 64_000;

function resolveCompactionReasoningSettings(
	config: Config,
	stored: ProviderSettings | undefined,
): ReasoningSettings | undefined {
	if (config.reasoningEffort) {
		return { enabled: true, effort: config.reasoningEffort };
	}
	return stored?.reasoning;
}

export function resolveCompactionProviderConfig(
	config: Config,
	providerSettingsManager: ProviderSettingsManager,
): ProviderConfig {
	const stored = providerSettingsManager.getProviderSettings(config.providerId);
	const providerConfig = toProviderConfig({
		...(stored ?? {}),
		provider: config.providerId,
		model: config.modelId,
		apiKey: config.apiKey || stored?.apiKey,
		baseUrl: config.baseUrl ?? stored?.baseUrl,
		headers: config.headers ?? stored?.headers,
		reasoning: resolveCompactionReasoningSettings(config, stored),
	} satisfies ProviderSettings);
	const base = {
		...providerConfig,
		...(config.providerConfig ?? {}),
	};
	return {
		...base,
		providerId: base.providerId ?? config.providerId,
		modelId: base.modelId ?? config.modelId,
		knownModels: base.knownModels ?? config.knownModels,
	};
}

export async function compactInteractiveMessages(input: {
	config: Config;
	providerSettingsManager: ProviderSettingsManager;
	sessionId: string;
	/** The canonical session transcript. */
	messages: Message[];
	/**
	 * The session's existing compacted working-context sidecar, if any. When
	 * it projects cleanly over `messages`, compaction runs on that projection
	 * — the transcript the model actually receives — instead of re-compacting
	 * the full canonical history. Canonical can outgrow the model's context
	 * window by millions of tokens on long sessions, and compacting it from
	 * scratch produces an over-window result that permanently wedges the
	 * session (cline/cline#12996).
	 */
	compactionState?: SessionCompactionState;
	abortSignal?: AbortSignal;
}): Promise<{
	compacted: boolean;
	canonicalMessages: Message[];
	compactionState?: SessionCompactionState;
}> {
	const modelInfo = input.config.knownModels?.[input.config.modelId];
	const compactionModelInfo = modelInfo
		? {
				...modelInfo,
				id: modelInfo.id ?? input.config.modelId,
			}
		: {
				id: input.config.modelId,
				maxInputTokens: FALLBACK_MANUAL_COMPACTION_MAX_INPUT_TOKENS,
			};
	const compact = createContextCompactionPrepareTurn(
		{
			providerConfig: resolveCompactionProviderConfig(
				input.config,
				input.providerSettingsManager,
			),
			providerId: input.config.providerId,
			modelId: input.config.modelId,
			compaction: {
				...input.config.compaction,
				enabled: true,
			},
			logger: input.config.logger,
			// Forward telemetry + sessionId so manual compactions emit
			// `task.compaction_executed` / `task.compaction_skipped` events
			// alongside auto compactions.
			telemetry: input.config.telemetry,
			sessionId: input.sessionId,
		},
		{ mode: "manual" },
	);
	if (!compact) {
		return { compacted: false, canonicalMessages: input.messages };
	}
	// Compact the working context — the same transcript every turn sends to
	// the model — not the raw canonical history. Summary-of-summary drift is
	// handled by the agentic strategy itself, which folds the previous
	// summary message forward instead of re-summarizing it blindly. Falls
	// back to canonical when there is no sidecar or it no longer projects.
	const projectedMessages = input.compactionState
		? projectSessionCompactionState(input.compactionState, input.messages)
		: undefined;
	const workingMessages = projectedMessages ?? input.messages;
	const result = await compact({
		agentId: "cli",
		conversationId: input.sessionId,
		parentAgentId: null,
		iteration: 0,
		messages: workingMessages,
		apiMessages: workingMessages,
		abortSignal: input.abortSignal ?? new AbortController().signal,
		systemPrompt: "",
		tools: [],
		model: {
			id: input.config.modelId,
			provider: input.config.providerId,
			info: compactionModelInfo,
		},
	});
	if (!result?.messages) {
		return { compacted: false, canonicalMessages: input.messages };
	}
	// The new sidecar stays keyed to the canonical transcript (same as the
	// SDK's own re-compaction flow), and a system prompt carried by the prior
	// sidecar survives unless this compaction rewrote it.
	const systemPrompt =
		result.systemPrompt ??
		(projectedMessages ? input.compactionState?.system_prompt : undefined);
	return {
		compacted: true,
		canonicalMessages: input.messages,
		compactionState: createSessionCompactionState({
			sourceMessages: input.messages,
			compactedMessages: result.messages,
			conversationId: input.sessionId,
			systemPrompt,
		}),
	};
}
