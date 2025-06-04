# 🤖🤝 Human-AI Collaboration

> **Advanced LLM orchestration tools and workflows for seamless Human-AI collaboration**

Welcome to the future of intelligent automation! This repository contains cutting-edge tools for orchestrating Large Language Models (LLMs) across multiple platforms and providers, designed for the collaborative partnership between humans and AI.

## 🌟 Features

### 🎯 LLM Orchestrator
- **Multi-Provider Support**: OpenAI, Anthropic, Azure OpenAI, and more
- **Chain LLM Calls**: Execute sequential LLM operations with context passing
- **Multi-Provider Queries**: Compare responses across different providers
- **Content Generation Pipeline**: Automated content creation with review cycles
- **Advanced Prompt Templating**: Dynamic templates with conditionals and loops
- **Context Management**: Preserve conversation state across operations

### 🔌 MCP Server Integration
- **Model Context Protocol**: Full MCP client implementation
- **Server Management**: Setup and manage MCP servers
- **Tool Execution**: Execute tools through MCP servers
- **Multi-Server Support**: Filesystem, database, web search, Git, Slack integrations

### 📋 Workflow Templates
- **Content Marketing Automation**: End-to-end content creation pipelines
- **Research Assistant**: MCP-enhanced research workflows
- **Documentation Generation**: Automated documentation with AI assistance
- **Best Practices**: Comprehensive guides and examples

## 🚀 Platform Support

This toolkit is designed to work seamlessly with:

- **[Activepieces](https://activepieces.com/)** - Native integration as custom pieces
- **[Flowise](https://flowiseai.com/)** - Import as custom nodes
- **[N8N](https://n8n.io/)** - Use as custom workflow components
- **Any workflow platform** - Modular design for easy adaptation

## 📁 Project Structure

```
Human-AI-Collaboration/
├── llm-orchestrator/           # Core LLM orchestration piece
│   ├── src/
│   │   ├── lib/
│   │   │   ├── actions/        # LLM actions and operations
│   │   │   ├── utils/          # Utility functions
│   │   │   └── types/          # TypeScript type definitions
│   │   └── index.ts            # Main piece export
│   ├── project.json            # NX project configuration
│   └── package.json            # Dependencies
├── mcp-server/                 # Model Context Protocol integration
│   ├── src/
│   │   ├── lib/
│   │   │   ├── actions/        # MCP actions
│   │   │   └── utils/          # MCP utilities
│   │   └── index.ts            # Main piece export
│   └── project.json            # NX project configuration
├── workflow-templates/         # Pre-built workflow templates
│   ├── llm-integration/        # LLM-focused workflows
│   ├── content-marketing/      # Marketing automation
│   └── research-assistant/     # Research workflows
└── docs/                       # Documentation and guides
```

## 🛠️ Installation & Setup

### For Activepieces

1. Copy the `llm-orchestrator` and `mcp-server` directories to your Activepieces pieces folder
2. Install dependencies:
   ```bash
   npm install @activepieces/pieces-framework openai @anthropic-ai/sdk @modelcontextprotocol/sdk
   ```
3. Build and deploy your pieces

### For Flowise

1. Import the workflow templates from `workflow-templates/`
2. Configure your API keys in the environment variables
3. Customize the flows according to your needs

### For N8N

1. Use the TypeScript code as custom function nodes
2. Configure HTTP request nodes for API calls
3. Import workflow templates and adapt as needed

## 🔧 Configuration

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# MCP Server Configuration
MCP_FILESYSTEM_ROOT=/path/to/filesystem/root
MCP_DATABASE_URL=your_database_connection_string
```

### API Keys Setup

1. **OpenAI**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Anthropic**: Get your API key from [Anthropic Console](https://console.anthropic.com/)
3. **Azure OpenAI**: Set up through [Azure Portal](https://portal.azure.com/)

## 📚 Usage Examples

### Simple LLM Call

```typescript
// Basic LLM interaction
const response = await simpleLLMCall({
  apiKey: "your-api-key",
  model: "gpt-4",
  systemPrompt: "You are a helpful assistant",
  userPrompt: "Explain quantum computing in simple terms",
  temperature: 0.7,
  maxTokens: 1000
});
```

### Chain LLM Calls

```typescript
// Sequential LLM operations with context
const pipeline = await chainLLMCalls({
  providers: [
    {
      name: "research",
      type: "openai",
      model: "gpt-4",
      prompt: "Research the topic: {{topic}}"
    },
    {
      name: "summarize",
      type: "anthropic",
      model: "claude-3-sonnet",
      prompt: "Summarize this research: {{research.content}}"
    }
  ],
  context: { topic: "AI in healthcare" }
});
```

### MCP Integration

```typescript
// Execute tools through MCP servers
const result = await mcpExecuteTool({
  serverUrl: "stdio://filesystem-server",
  toolName: "read_file",
  arguments: { path: "/path/to/file.txt" }
});
```

## 🎯 Workflow Templates

### Content Marketing Pipeline

Automated content creation workflow:
1. **Research Phase**: Gather information on topic
2. **Content Generation**: Create initial draft
3. **Review & Refinement**: AI-powered editing
4. **SEO Optimization**: Keyword optimization
5. **Final Review**: Human approval step

### Research Assistant

MCP-enhanced research workflow:
1. **Query Processing**: Understand research request
2. **Multi-Source Search**: Web, database, filesystem search
3. **Information Synthesis**: Combine and analyze findings
4. **Report Generation**: Create comprehensive report
5. **Citation Management**: Proper source attribution

## 🤝 Contributing

We believe in the power of Human-AI collaboration! Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- Built with ❤️ for the open-source community
- Inspired by the vision of seamless Human-AI collaboration
- Designed for the future of intelligent automation

## 🔗 Links

- [Activepieces Documentation](https://activepieces.com/docs)
- [Flowise Documentation](https://docs.flowiseai.com/)
- [N8N Documentation](https://docs.n8n.io/)
- [Model Context Protocol](https://modelcontextprotocol.io/)

---

**Ready to revolutionize your workflows? Let's build the future together! 🚀**