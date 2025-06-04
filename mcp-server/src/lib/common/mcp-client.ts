import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';

export interface MCPServerConfig {
  name: string;
  command: string;
  args?: string[];
  env?: Record<string, string>;
  timeout?: number;
}

export interface MCPTool {
  name: string;
  description: string;
  inputSchema: any;
}

export interface MCPToolCall {
  name: string;
  arguments: Record<string, any>;
}

export interface MCPToolResult {
  content: Array<{
    type: 'text' | 'image' | 'resource';
    text?: string;
    data?: string;
    mimeType?: string;
  }>;
  isError?: boolean;
}

export class MCPClient {
  private client: Client;
  private transport: StdioClientTransport;
  private connected: boolean = false;
  private tools: Map<string, MCPTool> = new Map();

  constructor(private config: MCPServerConfig) {
    this.transport = new StdioClientTransport({
      command: config.command,
      args: config.args || [],
      env: { ...process.env, ...config.env }
    });
    
    this.client = new Client(
      {
        name: `activepieces-mcp-client-${config.name}`,
        version: '1.0.0'
      },
      {
        capabilities: {
          tools: {}
        }
      }
    );
  }

  async connect(): Promise<void> {
    if (this.connected) {
      return;
    }

    try {
      await this.client.connect(this.transport);
      this.connected = true;
      
      // Load available tools
      await this.loadTools();
    } catch (error) {
      throw new Error(`Failed to connect to MCP server ${this.config.name}: ${error}`);
    }
  }

  async disconnect(): Promise<void> {
    if (!this.connected) {
      return;
    }

    try {
      await this.client.close();
      this.connected = false;
      this.tools.clear();
    } catch (error) {
      console.warn(`Error disconnecting from MCP server ${this.config.name}:`, error);
    }
  }

  private async loadTools(): Promise<void> {
    try {
      const response = await this.client.request(
        { method: 'tools/list' },
        ListToolsRequestSchema
      );

      this.tools.clear();
      for (const tool of response.tools) {
        this.tools.set(tool.name, {
          name: tool.name,
          description: tool.description || '',
          inputSchema: tool.inputSchema
        });
      }
    } catch (error) {
      throw new Error(`Failed to load tools from MCP server ${this.config.name}: ${error}`);
    }
  }

  async listTools(): Promise<MCPTool[]> {
    if (!this.connected) {
      await this.connect();
    }

    return Array.from(this.tools.values());
  }

  async getTool(name: string): Promise<MCPTool | undefined> {
    if (!this.connected) {
      await this.connect();
    }

    return this.tools.get(name);
  }

  async callTool(toolCall: MCPToolCall): Promise<MCPToolResult> {
    if (!this.connected) {
      await this.connect();
    }

    const tool = this.tools.get(toolCall.name);
    if (!tool) {
      throw new Error(`Tool ${toolCall.name} not found on MCP server ${this.config.name}`);
    }

    try {
      const response = await this.client.request(
        {
          method: 'tools/call',
          params: {
            name: toolCall.name,
            arguments: toolCall.arguments
          }
        },
        CallToolRequestSchema
      );

      return {
        content: response.content,
        isError: response.isError
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error calling tool ${toolCall.name}: ${error}`
        }],
        isError: true
      };
    }
  }

  async callMultipleTools(toolCalls: MCPToolCall[]): Promise<MCPToolResult[]> {
    const results: MCPToolResult[] = [];
    
    for (const toolCall of toolCalls) {
      try {
        const result = await this.callTool(toolCall);
        results.push(result);
      } catch (error) {
        results.push({
          content: [{
            type: 'text',
            text: `Error calling tool ${toolCall.name}: ${error}`
          }],
          isError: true
        });
      }
    }

    return results;
  }

  isConnected(): boolean {
    return this.connected;
  }

  getServerName(): string {
    return this.config.name;
  }

  async healthCheck(): Promise<boolean> {
    try {
      if (!this.connected) {
        await this.connect();
      }
      
      // Try to list tools as a health check
      await this.listTools();
      return true;
    } catch (error) {
      console.warn(`Health check failed for MCP server ${this.config.name}:`, error);
      return false;
    }
  }

  async refreshTools(): Promise<void> {
    if (!this.connected) {
      await this.connect();
    }

    await this.loadTools();
  }

  getToolsInfo(): { serverName: string; toolCount: number; tools: string[] } {
    return {
      serverName: this.config.name,
      toolCount: this.tools.size,
      tools: Array.from(this.tools.keys())
    };
  }
}

export class MCPServerManager {
  private clients: Map<string, MCPClient> = new Map();

  addServer(config: MCPServerConfig): void {
    const client = new MCPClient(config);
    this.clients.set(config.name, client);
  }

  removeServer(name: string): void {
    const client = this.clients.get(name);
    if (client) {
      client.disconnect();
      this.clients.delete(name);
    }
  }

  async getClient(name: string): Promise<MCPClient | undefined> {
    const client = this.clients.get(name);
    if (client && !client.isConnected()) {
      await client.connect();
    }
    return client;
  }

  async getAllTools(): Promise<{ server: string; tools: MCPTool[] }[]> {
    const allTools: { server: string; tools: MCPTool[] }[] = [];

    for (const [serverName, client] of this.clients) {
      try {
        const tools = await client.listTools();
        allTools.push({ server: serverName, tools });
      } catch (error) {
        console.warn(`Failed to get tools from server ${serverName}:`, error);
        allTools.push({ server: serverName, tools: [] });
      }
    }

    return allTools;
  }

  async findTool(toolName: string): Promise<{ server: string; tool: MCPTool } | undefined> {
    for (const [serverName, client] of this.clients) {
      try {
        const tool = await client.getTool(toolName);
        if (tool) {
          return { server: serverName, tool };
        }
      } catch (error) {
        console.warn(`Error searching for tool ${toolName} on server ${serverName}:`, error);
      }
    }

    return undefined;
  }

  async callTool(serverName: string, toolCall: MCPToolCall): Promise<MCPToolResult> {
    const client = await this.getClient(serverName);
    if (!client) {
      throw new Error(`MCP server ${serverName} not found`);
    }

    return await client.callTool(toolCall);
  }

  async healthCheckAll(): Promise<{ server: string; healthy: boolean }[]> {
    const results: { server: string; healthy: boolean }[] = [];

    for (const [serverName, client] of this.clients) {
      const healthy = await client.healthCheck();
      results.push({ server: serverName, healthy });
    }

    return results;
  }

  async disconnectAll(): Promise<void> {
    const disconnectPromises = Array.from(this.clients.values()).map(client => 
      client.disconnect()
    );

    await Promise.all(disconnectPromises);
  }

  listServers(): string[] {
    return Array.from(this.clients.keys());
  }

  getServerInfo(): { name: string; connected: boolean; toolCount: number; tools: string[] }[] {
    return Array.from(this.clients.entries()).map(([name, client]) => ({
      name,
      connected: client.isConnected(),
      ...client.getToolsInfo()
    }));
  }
}