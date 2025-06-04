import { createAction, Property } from '@activepieces/pieces-framework';
import { MCPServerManager, MCPServerConfig } from '../common/mcp-client';

// Global MCP server manager instance
const globalMCPManager = new MCPServerManager();

export const setupMCPServer = createAction({
  name: 'setup_mcp_server',
  displayName: 'Setup MCP Server',
  description: 'Configure and connect to a Model Context Protocol (MCP) server',
  props: {
    serverName: Property.ShortText({
      displayName: 'Server Name',
      required: true,
      description: 'Unique name for this MCP server'
    }),

    serverType: Property.StaticDropdown({
      displayName: 'Server Type',
      required: true,
      options: {
        options: [
          { label: 'File System', value: 'filesystem' },
          { label: 'Database', value: 'database' },
          { label: 'Web Search', value: 'web_search' },
          { label: 'API Tools', value: 'api_tools' },
          { label: 'Git Repository', value: 'git' },
          { label: 'Slack', value: 'slack' },
          { label: 'Google Drive', value: 'google_drive' },
          { label: 'Custom', value: 'custom' }
        ]
      },
      description: 'Type of MCP server to configure'
    }),

    command: Property.ShortText({
      displayName: 'Command',
      required: true,
      description: 'Command to start the MCP server (e.g., "npx", "python", "node")'
    }),

    args: Property.Array({
      displayName: 'Arguments',
      required: false,
      properties: {
        arg: Property.ShortText({
          displayName: 'Argument',
          required: true
        })
      },
      description: 'Command line arguments for the MCP server'
    }),

    environment: Property.Object({
      displayName: 'Environment Variables',
      required: false,
      description: 'Environment variables for the MCP server'
    }),

    timeout: Property.Number({
      displayName: 'Connection Timeout (seconds)',
      required: false,
      defaultValue: 30,
      description: 'Timeout for connecting to the MCP server'
    }),

    autoConnect: Property.Checkbox({
      displayName: 'Auto Connect',
      required: false,
      defaultValue: true,
      description: 'Automatically connect to the server after setup'
    }),

    testConnection: Property.Checkbox({
      displayName: 'Test Connection',
      required: false,
      defaultValue: true,
      description: 'Test the connection and list available tools'
    })
  },

  async run(context) {
    const { 
      serverName, 
      serverType, 
      command, 
      args, 
      environment, 
      timeout, 
      autoConnect, 
      testConnection 
    } = context.propsValue;

    try {
      // Prepare server configuration
      const serverConfig: MCPServerConfig = {
        name: serverName,
        command,
        args: args?.map(a => a.arg) || [],
        env: environment || {},
        timeout: timeout * 1000 // Convert to milliseconds
      };

      // Add predefined configurations for common server types
      switch (serverType) {
        case 'filesystem':
          if (!serverConfig.args.includes('@modelcontextprotocol/server-filesystem')) {
            serverConfig.args.unshift('@modelcontextprotocol/server-filesystem');
          }
          break;
        
        case 'database':
          if (!serverConfig.args.includes('@modelcontextprotocol/server-sqlite')) {
            serverConfig.args.unshift('@modelcontextprotocol/server-sqlite');
          }
          break;
        
        case 'web_search':
          if (!serverConfig.args.includes('@modelcontextprotocol/server-brave-search')) {
            serverConfig.args.unshift('@modelcontextprotocol/server-brave-search');
          }
          break;
        
        case 'git':
          if (!serverConfig.args.includes('@modelcontextprotocol/server-git')) {
            serverConfig.args.unshift('@modelcontextprotocol/server-git');
          }
          break;
        
        case 'slack':
          if (!serverConfig.args.includes('@modelcontextprotocol/server-slack')) {
            serverConfig.args.unshift('@modelcontextprotocol/server-slack');
          }
          break;
        
        case 'google_drive':
          if (!serverConfig.args.includes('@modelcontextprotocol/server-gdrive')) {
            serverConfig.args.unshift('@modelcontextprotocol/server-gdrive');
          }
          break;
      }

      // Add server to manager
      globalMCPManager.addServer(serverConfig);

      let connectionResult = null;
      let toolsInfo = null;

      // Auto connect if requested
      if (autoConnect) {
        const client = await globalMCPManager.getClient(serverName);
        if (!client) {
          throw new Error(`Failed to create client for server ${serverName}`);
        }

        connectionResult = {
          connected: client.isConnected(),
          serverName: client.getServerName()
        };

        // Test connection and get tools if requested
        if (testConnection) {
          const healthy = await client.healthCheck();
          if (!healthy) {
            throw new Error(`Health check failed for server ${serverName}`);
          }

          const tools = await client.listTools();
          toolsInfo = {
            toolCount: tools.length,
            tools: tools.map(tool => ({
              name: tool.name,
              description: tool.description,
              hasInputSchema: !!tool.inputSchema
            }))
          };
        }
      }

      return {
        success: true,
        serverName,
        serverType,
        configuration: {
          command: serverConfig.command,
          args: serverConfig.args,
          environment: Object.keys(serverConfig.env || {}),
          timeout: serverConfig.timeout
        },
        connection: connectionResult,
        tools: toolsInfo,
        message: `MCP server '${serverName}' configured successfully${autoConnect ? ' and connected' : ''}`
      };

    } catch (error) {
      // Clean up on error
      globalMCPManager.removeServer(serverName);
      
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        serverName,
        serverType
      };
    }
  }
});

// Export the global manager for use in other actions
export { globalMCPManager };