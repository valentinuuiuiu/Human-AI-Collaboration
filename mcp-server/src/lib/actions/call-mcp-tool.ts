import { createAction, Property } from '@activepieces/pieces-framework';
import { globalMCPManager } from './setup-mcp-server';
import { MCPToolCall } from '../common/mcp-client';

export const callMCPTool = createAction({
  name: 'call_mcp_tool',
  displayName: 'Call MCP Tool',
  description: 'Execute a tool from a connected MCP server',
  props: {
    serverName: Property.ShortText({
      displayName: 'Server Name',
      required: true,
      description: 'Name of the MCP server to use'
    }),

    toolName: Property.ShortText({
      displayName: 'Tool Name',
      required: true,
      description: 'Name of the tool to execute'
    }),

    arguments: Property.Object({
      displayName: 'Tool Arguments',
      required: false,
      description: 'Arguments to pass to the tool (JSON object)'
    }),

    autoDiscoverTool: Property.Checkbox({
      displayName: 'Auto Discover Tool',
      required: false,
      defaultValue: false,
      description: 'Automatically find the tool across all connected servers if not found on specified server'
    }),

    timeout: Property.Number({
      displayName: 'Execution Timeout (seconds)',
      required: false,
      defaultValue: 60,
      description: 'Maximum time to wait for tool execution'
    }),

    retryOnError: Property.Checkbox({
      displayName: 'Retry on Error',
      required: false,
      defaultValue: false,
      description: 'Retry the tool call once if it fails'
    }),

    includeMetadata: Property.Checkbox({
      displayName: 'Include Metadata',
      required: false,
      defaultValue: true,
      description: 'Include execution metadata in the response'
    })
  },

  async run(context) {
    const { 
      serverName, 
      toolName, 
      arguments: toolArgs, 
      autoDiscoverTool, 
      timeout, 
      retryOnError,
      includeMetadata 
    } = context.propsValue;

    const startTime = Date.now();

    try {
      let targetServer = serverName;
      let client = await globalMCPManager.getClient(serverName);

      // If client not found or tool not available, try auto-discovery
      if (!client || autoDiscoverTool) {
        const toolInfo = await globalMCPManager.findTool(toolName);
        if (toolInfo) {
          targetServer = toolInfo.server;
          client = await globalMCPManager.getClient(targetServer);
        }
      }

      if (!client) {
        throw new Error(`MCP server '${serverName}' not found or not connected`);
      }

      // Verify tool exists
      const tool = await client.getTool(toolName);
      if (!tool) {
        const availableTools = await client.listTools();
        throw new Error(
          `Tool '${toolName}' not found on server '${targetServer}'. ` +
          `Available tools: ${availableTools.map(t => t.name).join(', ')}`
        );
      }

      // Prepare tool call
      const toolCall: MCPToolCall = {
        name: toolName,
        arguments: toolArgs || {}
      };

      // Execute tool with timeout
      const executeWithTimeout = async () => {
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error('Tool execution timeout')), timeout * 1000);
        });

        const executionPromise = client!.callTool(toolCall);
        return Promise.race([executionPromise, timeoutPromise]);
      };

      let result;
      let retryAttempted = false;

      try {
        result = await executeWithTimeout();
      } catch (error) {
        if (retryOnError && !retryAttempted) {
          retryAttempted = true;
          console.warn(`Tool execution failed, retrying: ${error}`);
          
          // Wait a bit before retry
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          try {
            result = await executeWithTimeout();
          } catch (retryError) {
            throw new Error(`Tool execution failed after retry: ${retryError}`);
          }
        } else {
          throw error;
        }
      }

      const executionTime = Date.now() - startTime;

      // Process result content
      const processedContent = result.content.map(item => {
        switch (item.type) {
          case 'text':
            return {
              type: 'text',
              content: item.text || '',
              length: item.text?.length || 0
            };
          case 'image':
            return {
              type: 'image',
              mimeType: item.mimeType || 'image/png',
              data: item.data || '',
              size: item.data?.length || 0
            };
          case 'resource':
            return {
              type: 'resource',
              mimeType: item.mimeType || 'application/octet-stream',
              data: item.data || '',
              size: item.data?.length || 0
            };
          default:
            return {
              type: 'unknown',
              content: JSON.stringify(item)
            };
        }
      });

      // Extract text content for easy access
      const textContent = result.content
        .filter(item => item.type === 'text')
        .map(item => item.text)
        .join('\n');

      const response: any = {
        success: !result.isError,
        serverName: targetServer,
        toolName,
        content: processedContent,
        textContent,
        hasError: result.isError,
        contentTypes: [...new Set(result.content.map(item => item.type))],
        itemCount: result.content.length
      };

      if (includeMetadata) {
        response.metadata = {
          executionTime,
          retryAttempted,
          toolInfo: {
            name: tool.name,
            description: tool.description,
            hasInputSchema: !!tool.inputSchema
          },
          arguments: toolArgs,
          timestamp: new Date().toISOString()
        };
      }

      return response;

    } catch (error) {
      const executionTime = Date.now() - startTime;
      
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        serverName,
        toolName,
        arguments: toolArgs,
        metadata: includeMetadata ? {
          executionTime,
          timestamp: new Date().toISOString(),
          errorType: error instanceof Error ? error.constructor.name : 'UnknownError'
        } : undefined
      };
    }
  }
});