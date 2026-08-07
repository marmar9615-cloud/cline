import { describe, expect, it } from "vitest";
import {
	buildCollapsedPreview,
	buildReadFilesKeys,
	extractFullOutputText,
	parseReadFilesInput,
} from "./tool-parsing";

describe("buildReadFilesKeys", () => {
	it("produces unique keys when the same path is read twice", () => {
		const info = parseReadFilesInput({
			files: [{ path: "/a/SKILL.md" }, { path: "/a/SKILL.md" }],
		});
		const keys = buildReadFilesKeys(info?.files ?? []);

		expect(keys).toHaveLength(2);
		expect(new Set(keys).size).toBe(keys.length);
	});

	it("produces unique keys for duplicate paths from the file_paths shape", () => {
		const info = parseReadFilesInput({
			file_paths: ["/a/SKILL.md", "/a/SKILL.md", "/b/SKILL.md"],
		});
		const keys = buildReadFilesKeys(info?.files ?? []);

		expect(keys).toHaveLength(3);
		expect(new Set(keys).size).toBe(keys.length);
	});

	it("keeps distinct paths in unique keys", () => {
		const keys = buildReadFilesKeys([{ path: "/a.ts" }, { path: "/b.ts" }]);

		expect(new Set(keys).size).toBe(2);
	});

	it("returns no keys for an empty list", () => {
		expect(buildReadFilesKeys([])).toEqual([]);
	});
});

describe("extractFullOutputText", () => {
	it("returns strings unchanged", () => {
		expect(extractFullOutputText("hello\nworld")).toBe("hello\nworld");
	});

	it("unwraps built-in ToolOperationResult arrays", () => {
		expect(
			extractFullOutputText([
				{ result: "first" },
				{ result: [{ type: "text", text: "second" }] },
			]),
		).toBe("first\nsecond");
	});

	it("extracts text from the MCP CallToolResult shape with real newlines", () => {
		const raw = {
			content: [{ type: "text", text: "# Memory\n\nline one\nline two" }],
		};
		expect(extractFullOutputText(raw)).toBe("# Memory\n\nline one\nline two");
	});

	it("joins multiple MCP content parts and placeholders for non-text parts", () => {
		const raw = {
			content: [
				{ type: "text", text: "result A" },
				{ type: "image", data: "...", mimeType: "image/png" },
				{ type: "text", text: "result B" },
			],
			isError: false,
		};
		expect(extractFullOutputText(raw)).toBe("result A\n[image]\nresult B");
	});

	it("extracts embedded resource text from MCP content", () => {
		const raw = {
			content: [
				{
					type: "resource",
					resource: { uri: "file:///m.md", text: "resource body" },
				},
			],
		};
		expect(extractFullOutputText(raw)).toBe("resource body");
	});

	it("falls back to pretty JSON for objects without extractable content", () => {
		const raw = { structuredContent: { ok: true } };
		expect(extractFullOutputText(raw)).toBe(JSON.stringify(raw, null, 2));
	});
});

describe("buildCollapsedPreview", () => {
	it("keeps short text unchanged", () => {
		const preview = buildCollapsedPreview("one\ntwo", 5, 600);
		expect(preview.isLong).toBe(false);
		expect(preview.text).toBe("one\ntwo");
	});

	it("collapses on line count", () => {
		const text = ["1", "2", "3", "4", "5", "6", "7"].join("\n");
		const preview = buildCollapsedPreview(text, 5, 600);
		expect(preview.isLong).toBe(true);
		expect(preview.text).toBe("1\n2\n3\n4\n5\n... 2 more lines");
	});

	it("collapses a single long line on character count", () => {
		const text = "x".repeat(1000);
		const preview = buildCollapsedPreview(text, 5, 600);
		expect(preview.isLong).toBe(true);
		expect(preview.text).toBe(`${"x".repeat(600)}\n... 400 more chars`);
	});

	it("caps the preview length even when collapsing on lines", () => {
		const longLine = "y".repeat(1000);
		const text = [longLine, "a", "b", "c", "d", "e", "f"].join("\n");
		const preview = buildCollapsedPreview(text, 5, 600);
		expect(preview.isLong).toBe(true);
		expect(preview.text).toBe(`${"y".repeat(600)}\n... 2 more lines`);
	});

	it("uses singular wording for one hidden line", () => {
		const text = ["1", "2", "3", "4", "5", "6"].join("\n");
		const preview = buildCollapsedPreview(text, 5, 600);
		expect(preview.text.endsWith("... 1 more line")).toBe(true);
	});
});
