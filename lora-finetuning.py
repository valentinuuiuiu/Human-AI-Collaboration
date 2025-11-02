#!/usr/bin/env python3
"""
Human-AI Collaboration Assistant Training Pipeline
Creates specialized training data and prompt engineering for granite4:3b model
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
import ollama

class HumanAICollaborator:
    def __init__(self, base_model: str = "granite4:3b"):
        self.base_model = base_model
        self.training_data = []
        self.specialized_prompts = {}

        # Create directories
        os.makedirs("./training_data", exist_ok=True)
        os.makedirs("./prompts", exist_ok=True)

    def collect_real_conversations(self, conversation_history: List[Dict[str, Any]]):
        """Collect and analyze real conversation patterns"""

        patterns = {
            "code_review": [],
            "debugging": [],
            "architecture": [],
            "optimization": [],
            "testing": [],
            "documentation": []
        }

        for conversation in conversation_history:
            if conversation.get("role") == "user":
                content = conversation.get("content", "").lower()

                # Categorize conversations
                if any(word in content for word in ["review", "code review", "pr", "pull request"]):
                    patterns["code_review"].append(conversation)
                elif any(word in content for word in ["bug", "error", "fix", "debug"]):
                    patterns["debugging"].append(conversation)
                elif any(word in content for word in ["architecture", "design", "structure"]):
                    patterns["architecture"].append(conversation)
                elif any(word in content for word in ["optimize", "performance", "speed"]):
                    patterns["optimization"].append(conversation)
                elif any(word in content for word in ["test", "testing", "unit test"]):
                    patterns["testing"].append(conversation)
                elif any(word in content for word in ["doc", "document", "readme"]):
                    patterns["documentation"].append(conversation)

        return patterns

    def generate_specialized_prompts(self, topics: List[str]):
        """Generate specialized prompts for different collaboration scenarios"""

        for topic in topics:
            prompt_template = f"""You are an expert Human-AI collaboration assistant specializing in {topic}.

Your role is to:
- Provide clear, actionable guidance
- Ask clarifying questions when needed
- Suggest best practices and patterns
- Explain technical concepts simply
- Help with implementation details
- Review and improve code/solutions

When collaborating on {topic}, focus on:
- Understanding the user's goals
- Breaking down complex problems
- Providing step-by-step solutions
- Explaining your reasoning
- Suggesting alternatives when appropriate

Always maintain a helpful, professional, and collaborative tone."""

            self.specialized_prompts[topic] = prompt_template

        return self.specialized_prompts

    def create_training_examples(self, topics: List[str], num_examples: int = 5):
        """Create training examples for each topic using the base model"""

        training_examples = {}

        for topic in topics:
            examples = []

            for i in range(num_examples):
                prompt = f"""Generate a realistic Human-AI collaboration example about {topic}.

Create a conversation where:
- User asks for help with a {topic} task
- AI provides helpful, step-by-step guidance
- Shows collaborative problem-solving

Format as JSON:
{{
  "user_query": "user's question",
  "ai_response": "AI's helpful response",
  "topic": "{topic}",
  "collaboration_style": "step-by-step guidance"
}}"""

                try:
                    response = ollama.chat(
                        model=self.base_model,
                        messages=[{"role": "user", "content": prompt}],
                        format="json"
                    )

                    if response and "message" in response:
                        content = response["message"].get("content", "")
                        try:
                            example = json.loads(content)
                            examples.append(example)
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON for {topic} example {i}")

                except Exception as e:
                    print(f"Error generating example for {topic}: {e}")

            training_examples[topic] = examples

        self.training_data = training_examples
        return training_examples

    def save_training_data(self):
        """Save the generated training data"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./training_data/training_examples_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump({
                "training_examples": self.training_data,
                "specialized_prompts": self.specialized_prompts,
                "metadata": {
                    "model": self.base_model,
                    "generated_at": timestamp,
                    "topics_covered": list(self.training_data.keys()) if isinstance(self.training_data, dict) else []
                }
            }, f, indent=2)

        print(f"💾 Training data saved to: {filename}")
        return filename

    def create_collaboration_guide(self):
        """Create a comprehensive collaboration guide"""

        guide = {
            "introduction": "This guide helps you collaborate effectively with AI assistants on software development tasks.",
            "best_practices": [
                "Be specific about your goals and constraints",
                "Provide context about your codebase and technology stack",
                "Break complex tasks into smaller, manageable steps",
                "Ask for clarification when suggestions are unclear",
                "Review and test AI-generated code before using in production",
                "Use AI for learning and understanding, not just code generation"
            ],
            "collaboration_patterns": {
                "code_review": "Focus on logic, edge cases, and best practices",
                "debugging": "Describe symptoms, provide error messages, share relevant code",
                "architecture": "Explain requirements, constraints, and success criteria",
                "optimization": "Share performance metrics and bottlenecks",
                "testing": "Define test scenarios and expected behaviors",
                "documentation": "Specify audience, purpose, and key information to include"
            },
            "prompt_engineering_tips": [
                "Start with clear problem statements",
                "Include relevant context and constraints",
                "Specify output format when needed",
                "Ask for explanations of complex solutions",
                "Request alternatives and trade-offs",
                "Use follow-up questions to refine understanding"
            ]
        }

        guide_filename = "./prompts/collaboration_guide.json"
        with open(guide_filename, 'w') as f:
            json.dump(guide, f, indent=2)

        print(f"📚 Collaboration guide saved to: {guide_filename}")
        return guide

def create_collaboration_topics():
    """Define the key topics for Human-AI collaboration"""

    return [
        "code review and optimization",
        "TypeScript development best practices",
        "LLM orchestration patterns",
        "MCP server integration",
        "error handling in AI systems",
        "workflow automation",
        "performance benchmarking",
        "testing strategies for AI applications",
        "Human-AI collaboration principles",
        "local LLM deployment and management"
    ]

if __name__ == "__main__":
    # Initialize the collaborator
    collaborator = HumanAICollaborator()

    # Define collaboration topics
    topics = create_collaboration_topics()

    print("🎯 Generating specialized prompts...")
    prompts = collaborator.generate_specialized_prompts(topics)

    print("📝 Creating training examples...")
    examples = collaborator.create_training_examples(topics, num_examples=3)

    print("📚 Creating collaboration guide...")
    guide = collaborator.create_collaboration_guide()

    print("💾 Saving training data...")
    collaborator.save_training_data()

    print("🎉 Human-AI collaboration assistant training complete!")
    print("🤖 Your specialized assistant is ready for effective collaboration!")