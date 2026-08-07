import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type CoreCompactionContext,
	createSessionCompactionState,
	ProviderSettingsManager,
	projectSessionCompactionState,
} from "@cline/core";
import type { Message } from "@cline/shared";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Config } from "../../utils/types";
import {
	compactInteractiveMessages,
	resolveCompactionProviderConfig,
} from "./compaction";

const createHandlerMock = vi.fn();

// Core defaults to the agentic compaction strategy, which summarizes via a
// real LLM handler. Stub only `createHandlerAsync` so no network call (or API
// key) is needed; every other `@cline/llms` export stays real because
// `@cline/core` re-exports them.
vi.mock("@cline/llms", async (importOriginal) => ({
	...(await importOriginal<typeof import("@cline/llms")>()),
	createHandlerAsync: (config: unknown) => createHandlerMock(config),
}));

async function* streamChunks(
	chunks: Array<Record<string, unknown>>,
): AsyncGenerator<Record<string, unknown>> {
	for (const chunk of chunks) {
		yield chunk;
	}
}

function createConfig(): Config {
	return {
		providerId: "anthropic",
		modelId: "claude-test",
		apiKey: "",
		cwd: "/tmp/project",
		workspaceRoot: "/tmp/project",
		systemPrompt: "system",
		mode: "act",
		enableTools: true,
		enableSpawnAgent: true,
		enableAgentTeams: true,
		verbose: false,
		thinking: false,
		outputMode: "text",
		sandbox: false,
		defaultToolAutoApprove: true,
		toolPolicies: {
			"*": { autoApprove: true },
		},
	};
}

const providerSettingsTempDirs: string[] = [];

function createProviderSettingsManager(): ProviderSettingsManager {
	const tempDir = mkdtempSync(join(tmpdir(), "cline-cli-compact-"));
	providerSettingsTempDirs.push(tempDir);
	return new ProviderSettingsManager({
		filePath: join(tempDir, "providers.json"),
	});
}

afterEach(() => {
	createHandlerMock.mockReset();
	for (const tempDir of providerSettingsTempDirs.splice(0)) {
		rmSync(tempDir, { force: true, recursive: true });
	}
});

describe("compactInteractiveMessages", () => {
	it("resolves manual compaction provider config from persisted OAuth settings", () => {
		const manager = createProviderSettingsManager();
		manager.saveProviderSettings({
			provider: "openai-native",
			model: "old-model",
			auth: {
				accessToken: "stored-access-token",
				refreshToken: "stored-refresh-token",
				accountId: "acct-1",
			},
			baseUrl: "https://stored.example.com/v1",
			headers: {
				"x-stored": "yes",
			},
		});
		const config = createConfig();
		config.providerId = "openai-native";
		config.modelId = "gpt-test";

		const providerConfig = resolveCompactionProviderConfig(config, manager);

		expect(providerConfig.providerId).toBe("openai-native");
		expect(providerConfig.modelId).toBe("gpt-test");
		expect(providerConfig.apiKey).toBe("stored-access-token");
		expect(providerConfig.accessToken).toBe("stored-access-token");
		expect(providerConfig.refreshToken).toBe("stored-refresh-token");
		expect(providerConfig.accountId).toBe("acct-1");
		expect(providerConfig.baseUrl).toBe("https://stored.example.com/v1");
		expect(providerConfig.headers).toEqual({ "x-stored": "yes" });
	});

	it("prefers active CLI reasoning effort over persisted provider reasoning settings", () => {
		const manager = createProviderSettingsManager();
		manager.saveProviderSettings({
			provider: "anthropic",
			model: "old-model",
			reasoning: { enabled: true, effort: "low" },
		});
		const config = createConfig();
		config.reasoningEffort = "high";

		const providerConfig = resolveCompactionProviderConfig(config, manager);

		expect(providerConfig.reasoningEffort).toBe("high");
	});

	it("passes the selected model context window to manual compaction", async () => {
		const longText = "x".repeat(16_000);
		const messages = Array.from({ length: 10 }, (_, index) => ({
			role: index % 2 === 0 ? ("user" as const) : ("assistant" as const),
			content: `message ${index} ${longText}`,
		}));
		const config = createConfig();
		const compact = vi.fn((context: CoreCompactionContext) => {
			expect(context.budget.request.maxInputTokens).toBe(400_000);
			return { messages: [messages[0]] };
		});
		config.knownModels = {
			"claude-test": {
				id: "claude-test",
				maxInputTokens: 400_000,
			},
		};
		config.compaction = { compact };

		const result = await compactInteractiveMessages({
			config,
			providerSettingsManager: createProviderSettingsManager(),
			sessionId: "sess-compact",
			messages,
		});

		expect(compact).toHaveBeenCalledTimes(1);
		expect(result.compacted).toBe(true);
		expect(result.canonicalMessages).toEqual(messages);
		expect(result.compactionState?.messages).toEqual([messages[0]]);
	});

	it("uses 90 percent of legacy contextWindow for manual compaction", async () => {
		const longText = "x".repeat(16_000);
		const messages = Array.from({ length: 10 }, (_, index) => ({
			role: index % 2 === 0 ? ("user" as const) : ("assistant" as const),
			content: `message ${index} ${longText}`,
		}));
		const config = createConfig();
		const compact = vi.fn((context: CoreCompactionContext) => {
			expect(context.budget.request.maxInputTokens).toBe(360_000);
			return { messages: [messages[0]] };
		});
		config.knownModels = {
			"claude-test": {
				id: "claude-test",
				contextWindow: 400_000,
			},
		};
		config.compaction = { compact };

		const result = await compactInteractiveMessages({
			config,
			providerSettingsManager: createProviderSettingsManager(),
			sessionId: "sess-compact",
			messages,
		});

		expect(compact).toHaveBeenCalledTimes(1);
		expect(result.compacted).toBe(true);
		expect(result.canonicalMessages).toEqual(messages);
		expect(result.compactionState?.messages).toEqual([messages[0]]);
	});

	it("uses a useful target budget for manual compaction", async () => {
		const mockSummary = "## Goal\nMocked agentic compaction summary";
		createHandlerMock.mockReturnValue({
			createMessage: vi.fn(() =>
				streamChunks([
					{ type: "text", id: "summary-1", text: mockSummary },
					{ type: "done", id: "summary-1", success: true },
				]),
			),
		});
		const longText = "x".repeat(16_000);
		const messages = Array.from({ length: 10 }, (_, index) => ({
			role: index % 2 === 0 ? ("user" as const) : ("assistant" as const),
			content: `message ${index} ${longText}`,
		}));

		const result = await compactInteractiveMessages({
			config: createConfig(),
			providerSettingsManager: createProviderSettingsManager(),
			sessionId: "sess-compact",
			messages,
		});

		const compactedMessages = result.compactionState?.messages ?? [];
		const compactedTextLength = compactedMessages.reduce(
			(total, message) =>
				total +
				(typeof message.content === "string" ? message.content.length : 0),
			0,
		);

		expect(result.compacted).toBe(true);
		expect(result.canonicalMessages).toEqual(messages);
		expect(compactedMessages.length).toBeGreaterThan(1);
		expect(compactedMessages.length).toBeLessThan(messages.length);
		expect(compactedTextLength).toBeGreaterThan(1_000);

		// The agentic strategy folds older messages into a summary message
		// built from the (mocked) summarizer output.
		expect(createHandlerMock).toHaveBeenCalledTimes(1);
		const [summaryMessage] = compactedMessages;
		const summaryText = Array.isArray(summaryMessage?.content)
			? summaryMessage.content
					.map((block) => ("text" in block ? block.text : ""))
					.join("\n")
			: String(summaryMessage?.content ?? "");
		expect(summaryText).toContain(mockSummary);
	});

	it("compacts the projected working context when a sidecar is provided", async () => {
		// cline/cline#12996: manual compaction must run over the working
		// context (existing sidecar projected over canonical), not the full
		// canonical history, which can exceed the model window by millions of
		// tokens on long sessions.
		const canonical: Message[] = [
			{ role: "user", content: "huge old prefix" },
			{ role: "assistant", content: "huge old reply" },
			{ role: "user", content: "new tail message" },
		];
		const priorSummary: Message = { role: "user", content: "prior summary" };
		const sidecar = createSessionCompactionState({
			sourceMessages: canonical.slice(0, 2),
			compactedMessages: [priorSummary],
			conversationId: "sess-compact",
			systemPrompt: "carried system prompt",
		});
		const config = createConfig();
		const seenTranscripts: Message[][] = [];
		config.compaction = {
			compact: (context: CoreCompactionContext) => {
				seenTranscripts.push(context.messages as Message[]);
				return { messages: [{ role: "user", content: "fresh summary" }] };
			},
		};

		const result = await compactInteractiveMessages({
			config,
			providerSettingsManager: createProviderSettingsManager(),
			sessionId: "sess-compact",
			messages: canonical,
			compactionState: sidecar,
		});

		// The compactor saw the projection (prior summary + canonical tail),
		// not the huge canonical prefix.
		expect(seenTranscripts).toEqual([[priorSummary, canonical[2]]]);
		expect(result.compacted).toBe(true);
		// The new sidecar is keyed to canonical and projects cleanly over it,
		// and the prior sidecar's system prompt survives.
		expect(result.compactionState?.source_message_count).toBe(canonical.length);
		expect(result.compactionState?.system_prompt).toBe("carried system prompt");
		expect(
			result.compactionState &&
				projectSessionCompactionState(result.compactionState, canonical),
		).toEqual([{ role: "user", content: "fresh summary" }]);
	});

	it("falls back to the canonical transcript when the sidecar no longer projects", async () => {
		const canonical: Message[] = [
			{ role: "user", content: "message one" },
			{ role: "assistant", content: "message two" },
		];
		// A sidecar keyed to a different history: projection fails.
		const staleSidecar = createSessionCompactionState({
			sourceMessages: [{ role: "user", content: "some other history" }],
			compactedMessages: [{ role: "user", content: "stale summary" }],
			conversationId: "sess-compact",
		});
		const config = createConfig();
		const seenTranscripts: Message[][] = [];
		config.compaction = {
			compact: (context: CoreCompactionContext) => {
				seenTranscripts.push(context.messages as Message[]);
				return { messages: [{ role: "user", content: "fresh summary" }] };
			},
		};

		const result = await compactInteractiveMessages({
			config,
			providerSettingsManager: createProviderSettingsManager(),
			sessionId: "sess-compact",
			messages: canonical,
			compactionState: staleSidecar,
		});

		expect(seenTranscripts).toEqual([canonical]);
		expect(result.compacted).toBe(true);
	});

	it("reports compaction when core returns changed messages with the same count", async () => {
		const messages = [
			{
				role: "user" as const,
				content: `${" ".repeat(80)}same count but content should be trimmed${" ".repeat(80)}`,
			},
		];
		const config = createConfig();
		config.compaction = {
			compact: () => ({
				messages: [
					{
						role: "user",
						content: "same count but content should be trimmed",
					},
				],
			}),
		};

		const result = await compactInteractiveMessages({
			config,
			providerSettingsManager: createProviderSettingsManager(),
			sessionId: "sess-compact",
			messages,
		});

		expect(result.compacted).toBe(true);
		expect(result.canonicalMessages).toEqual(messages);
		expect(result.compactionState?.messages).toHaveLength(messages.length);
		expect(result.compactionState?.messages[0]?.content).toBe(
			"same count but content should be trimmed",
		);
	});
});
