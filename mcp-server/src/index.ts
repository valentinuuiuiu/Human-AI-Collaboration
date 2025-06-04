import { createPiece, PieceAuth } from '@activepieces/pieces-framework';
import { PieceCategory } from '@activepieces/shared';
import { setupMCPServer } from './lib/actions/setup-mcp-server';
import { callMCPTool } from './lib/actions/call-mcp-tool';

export const mcpServerAuth = PieceAuth.None();

export const mcpServer = createPiece({
  displayName: 'MCP Server',
  description: 'Model Context Protocol (MCP) server integration for enhanced LLM tool capabilities',
  minimumSupportedRelease: '0.36.1',
  logoUrl: 'https://cdn.activepieces.com/pieces/mcp-server.png',
  categories: [PieceCategory.ARTIFICIAL_INTELLIGENCE, PieceCategory.DEVELOPER_TOOLS],
  auth: mcpServerAuth,
  actions: [
    setupMCPServer,
    callMCPTool,
  ],
  authors: ['openhands'],
  triggers: [],
});