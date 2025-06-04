# 🚀 Getting Started with Human-AI Collaboration

Welcome to the future of intelligent automation! This guide will help you get up and running with our LLM orchestration tools.

## 🎯 Quick Start

### 1. Choose Your Platform

Our tools work with multiple workflow platforms:

- **Activepieces** - Native integration (recommended for beginners)
- **Flowise** - Visual workflow builder
- **N8N** - Open-source automation platform
- **Custom Integration** - Use our TypeScript modules directly

### 2. Basic Setup

#### For Activepieces Users

1. **Clone this repository**:
   ```bash
   git clone https://github.com/valentinuuiuiu/Human-AI-Collaboration.git
   cd Human-AI-Collaboration
   ```

2. **Install dependencies**:
   ```bash
   npm run install:all
   ```

3. **Copy pieces to your Activepieces installation**:
   ```bash
   cp -r llm-orchestrator /path/to/activepieces/packages/pieces/community/
   cp -r mcp-server /path/to/activepieces/packages/pieces/community/
   ```

4. **Build the pieces**:
   ```bash
   cd /path/to/activepieces
   npm run build
   ```

#### For Flowise Users

1. **Import workflow templates**:
   - Open Flowise
   - Go to "Chatflows"
   - Click "Import"
   - Select files from `workflow-templates/flowise/`

2. **Configure API keys**:
   - Set environment variables or use Flowise's credential system
   - Add your OpenAI, Anthropic, or Azure OpenAI keys

#### For N8N Users

1. **Import workflows**:
   - Open N8N
   - Go to "Workflows"
   - Click "Import from file"
   - Select files from `workflow-templates/n8n/`

2. **Configure credentials**:
   - Set up HTTP Request credentials for API calls
   - Configure environment variables

### 3. Get Your API Keys

#### OpenAI
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Set `OPENAI_API_KEY` environment variable

#### Anthropic
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Create a new API key
3. Set `ANTHROPIC_API_KEY` environment variable

#### Azure OpenAI
1. Set up Azure OpenAI service in [Azure Portal](https://portal.azure.com/)
2. Get your endpoint and API key
3. Set `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` environment variables

## 🎮 Your First Workflow

Let's create a simple content generation workflow:

### Step 1: Simple LLM Call

```typescript
// Basic AI interaction
const response = await simpleLLMCall({
  apiKey: process.env.OPENAI_API_KEY,
  model: "gpt-4",
  systemPrompt: "You are a creative writing assistant",
  userPrompt: "Write a short story about a robot learning to paint",
  temperature: 0.8,
  maxTokens: 500
});

console.log(response.result.content);
```

### Step 2: Chain Multiple LLM Calls

```typescript
// Create a content pipeline
const pipeline = await chainLLMCalls({
  providers: [
    {
      name: "ideation",
      type: "openai",
      model: "gpt-4",
      prompt: "Generate 3 blog post ideas about {{topic}}"
    },
    {
      name: "outline",
      type: "anthropic", 
      model: "claude-3-sonnet",
      prompt: "Create a detailed outline for this blog post idea: {{ideation.content}}"
    },
    {
      name: "content",
      type: "openai",
      model: "gpt-4",
      prompt: "Write a full blog post based on this outline: {{outline.content}}"
    }
  ],
  context: { topic: "AI in everyday life" }
});
```

### Step 3: Add MCP Integration

```typescript
// Research with MCP servers
const research = await mcpExecuteTool({
  serverUrl: "stdio://web-search-server",
  toolName: "search",
  arguments: { 
    query: "latest AI trends 2024",
    maxResults: 5 
  }
});

// Use research in content generation
const informedContent = await simpleLLMCall({
  apiKey: process.env.OPENAI_API_KEY,
  model: "gpt-4",
  systemPrompt: "You are a tech journalist",
  userPrompt: `Write an article about AI trends based on this research: ${research.content}`,
  temperature: 0.7,
  maxTokens: 1000
});
```

## 🛠️ Advanced Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# LLM Providers
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# MCP Servers
MCP_FILESYSTEM_ROOT=/path/to/your/files
MCP_DATABASE_URL=postgresql://user:pass@localhost:5432/db
MCP_WEB_SEARCH_API_KEY=your-search-api-key

# Workflow Configuration
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=1000
ENABLE_USAGE_TRACKING=true
```

### Custom Providers

Add your own LLM providers:

```typescript
// In llm-orchestrator/src/lib/utils/providers.ts
export const customProvider = {
  name: 'custom',
  createClient: (config: any) => {
    return new CustomLLMClient(config);
  },
  models: ['custom-model-1', 'custom-model-2']
};
```

## 🎯 Workflow Templates

### Content Marketing Pipeline

Perfect for automated content creation:

1. **Topic Research** - Gather information about your topic
2. **Content Planning** - Create content calendar and outlines  
3. **Content Generation** - Write articles, social posts, etc.
4. **SEO Optimization** - Optimize for search engines
5. **Review & Publishing** - Human review and approval

### Research Assistant

Enhanced with MCP servers:

1. **Query Processing** - Understand research requirements
2. **Multi-Source Search** - Web, databases, files
3. **Information Synthesis** - Combine and analyze findings
4. **Report Generation** - Create comprehensive reports
5. **Citation Management** - Proper source attribution

### Customer Support Automation

AI-powered support workflows:

1. **Ticket Classification** - Categorize support requests
2. **Knowledge Base Search** - Find relevant solutions
3. **Response Generation** - Create helpful responses
4. **Escalation Logic** - Route complex issues to humans
5. **Follow-up Management** - Automated follow-ups

## 🔧 Troubleshooting

### Common Issues

#### "API Key not found"
- Check your environment variables
- Ensure `.env` file is in the correct location
- Verify API key format and permissions

#### "Model not available"
- Check if you have access to the specified model
- Verify your API subscription level
- Try a different model (e.g., gpt-3.5-turbo instead of gpt-4)

#### "Rate limit exceeded"
- Implement retry logic with exponential backoff
- Consider using multiple API keys
- Reduce request frequency

#### "MCP server connection failed"
- Check MCP server configuration
- Verify server is running and accessible
- Check network connectivity and permissions

### Getting Help

1. **Check the documentation** in the `docs/` folder
2. **Review workflow templates** for examples
3. **Open an issue** on GitHub for bugs or feature requests
4. **Join the community** discussions

## 🎉 Next Steps

Now that you're set up, explore:

1. **Advanced Workflows** - Check out complex templates
2. **Custom Integrations** - Build your own pieces
3. **Performance Optimization** - Fine-tune for your use case
4. **Community Contributions** - Share your workflows

Happy automating! 🚀