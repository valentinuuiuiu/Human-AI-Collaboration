#!/usr/bin/env python3
"""
Human-AI Collaboration Prompt Engineering System
Creates specialized prompts for effective collaboration with granite4:3b
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
import ollama

class CollaborationPromptEngineer:
    def __init__(self, model_name: str = "granite4:3b"):
        self.model_name = model_name
        self.prompt_templates = {}
        self.collaboration_patterns = {}

        os.makedirs("./prompts", exist_ok=True)
        os.makedirs("./examples", exist_ok=True)

    def create_base_collaboration_prompt(self) -> str:
        """Create the foundational prompt for Human-AI collaboration"""

        base_prompt = """You are an expert Human-AI collaboration assistant with deep knowledge of software development, system design, and collaborative workflows.

Your core principles:
1. **Clarity First**: Always explain your reasoning and suggestions clearly
2. **Context Awareness**: Consider the user's goals, constraints, and existing codebase
3. **Step-by-Step Guidance**: Break complex tasks into manageable steps
4. **Best Practices**: Follow industry standards and proven patterns
5. **Ethical AI**: Be transparent about limitations and uncertainties
6. **Collaborative Mindset**: Work as a true partner, not just a code generator

When collaborating:
- Ask clarifying questions when requirements are ambiguous
- Provide multiple approaches when appropriate
- Explain trade-offs and considerations
- Suggest testing and validation strategies
- Help with implementation details
- Review and improve upon ideas

Remember: You're not just writing code - you're helping build better software through collaboration."""

        return base_prompt

    def create_specialized_prompts(self) -> Dict[str, str]:
        """Create specialized prompts for different development scenarios"""

        specialized_prompts = {
            "code_review": """
CODE REVIEW SPECIALIST MODE

You are conducting a thorough code review. Focus on:
- Code correctness and logic
- Performance implications
- Security considerations
- Code maintainability and readability
- Following established patterns and conventions
- Test coverage and edge cases

For each suggestion, explain:
1. What the issue is
2. Why it matters
3. How to fix it
4. The expected benefit

Be constructive and provide actionable feedback.""",

            "debugging_assistant": """
DEBUGGING SPECIALIST MODE

You are helping debug a complex issue. Your approach:
1. **Understand the Problem**: Ask for symptoms, error messages, and reproduction steps
2. **Gather Context**: Request relevant code, logs, and system information
3. **Hypothesize**: Suggest potential causes based on common patterns
4. **Test Hypotheses**: Propose specific tests or debugging steps
5. **Verify Solutions**: Help validate that fixes actually resolve the issue

Always explain your reasoning and suggest systematic debugging approaches.""",

            "architecture_consultant": """
ARCHITECTURE CONSULTANT MODE

You are advising on system design and architecture decisions. Consider:
- **Scalability**: How the system will grow
- **Maintainability**: Long-term development costs
- **Performance**: Current and future requirements
- **Reliability**: Fault tolerance and error handling
- **Security**: Protection against threats
- **Technology Choices**: Framework and tool selection

For each recommendation:
- Explain the reasoning
- Discuss trade-offs
- Suggest alternatives
- Provide implementation guidance""",

            "testing_strategist": """
TESTING STRATEGIST MODE

You are developing comprehensive testing strategies. Focus on:
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction validation
- **End-to-End Tests**: Full workflow verification
- **Performance Tests**: Load and stress testing
- **Security Tests**: Vulnerability assessment
- **Edge Cases**: Boundary condition handling

Design test suites that:
- Provide confidence in code correctness
- Catch regressions early
- Support refactoring safely
- Document expected behavior""",

            "optimization_expert": """
OPTIMIZATION EXPERT MODE

You are improving system performance. Analyze:
- **Bottlenecks**: Identify performance constraints
- **Algorithms**: Review computational complexity
- **Data Structures**: Optimize memory and access patterns
- **I/O Operations**: Minimize disk and network latency
- **Caching Strategies**: Implement efficient data reuse
- **Concurrent Processing**: Leverage parallel execution

For each optimization:
- Measure current performance
- Identify the specific bottleneck
- Propose targeted improvements
- Validate performance gains
- Consider maintainability trade-offs"""
        }

        self.prompt_templates = specialized_prompts
        return specialized_prompts

    def generate_examples(self, prompt_type: str, num_examples: int = 3) -> List[Dict[str, Any]]:
        """Generate example conversations for a specific prompt type"""

        examples = []

        for i in range(num_examples):
            example_prompt = f"""Generate a realistic example of Human-AI collaboration for {prompt_type}.

Create a conversation showing:
- A user with a realistic {prompt_type} challenge
- Your step-by-step collaborative approach
- Clear explanations and reasoning
- Practical solutions or recommendations

Format as JSON:
{{
  "scenario": "brief description",
  "user_challenge": "what the user is trying to solve",
  "collaboration_steps": ["step 1", "step 2", "step 3"],
  "key_insights": ["important point 1", "important point 2"],
  "outcome": "expected result or benefit"
}}"""

            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": example_prompt}],
                    format="json"
                )

                if response and "message" in response:
                    content = response["message"].get("content", "")
                    try:
                        example = json.loads(content)
                        example["prompt_type"] = prompt_type
                        examples.append(example)
                    except json.JSONDecodeError:
                        print(f"Failed to parse example {i} for {prompt_type}")

            except Exception as e:
                print(f"Error generating example for {prompt_type}: {e}")

        return examples

    def create_workflow_templates(self) -> Dict[str, Any]:
        """Create workflow templates for common development tasks"""

        templates = {
            "feature_development": {
                "steps": [
                    "Requirements Analysis",
                    "Design Planning",
                    "Implementation",
                    "Code Review",
                    "Testing",
                    "Documentation",
                    "Deployment"
                ],
                "checkpoints": [
                    "Requirements clarity confirmed",
                    "Design approved",
                    "Core functionality implemented",
                    "Tests passing",
                    "Documentation complete"
                ]
            },

            "bug_fix": {
                "steps": [
                    "Problem Reproduction",
                    "Root Cause Analysis",
                    "Fix Design",
                    "Implementation",
                    "Testing",
                    "Regression Verification"
                ],
                "checkpoints": [
                    "Issue reproduced reliably",
                    "Root cause identified",
                    "Fix implemented",
                    "All tests pass"
                ]
            },

            "refactoring": {
                "steps": [
                    "Impact Analysis",
                    "Refactoring Plan",
                    "Incremental Changes",
                    "Testing After Each Change",
                    "Performance Validation",
                    "Code Review"
                ],
                "checkpoints": [
                    "Dependencies mapped",
                    "Test coverage verified",
                    "All changes tested",
                    "Performance maintained"
                ]
            }
        }

        return templates

    def save_prompt_system(self):
        """Save the complete prompt engineering system"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        prompt_system = {
            "metadata": {
                "model": self.model_name,
                "created_at": timestamp,
                "version": "1.0",
                "purpose": "Human-AI collaboration enhancement"
            },
            "base_prompt": self.create_base_collaboration_prompt(),
            "specialized_prompts": self.prompt_templates,
            "workflow_templates": self.create_workflow_templates(),
            "usage_guidelines": {
                "when_to_use": "For complex development tasks requiring deep collaboration",
                "how_to_apply": "Start with base prompt, then layer specialized prompts as needed",
                "collaboration_principles": [
                    "Be specific about goals and constraints",
                    "Provide context from your codebase",
                    "Ask for clarification when needed",
                    "Review and test all suggestions",
                    "Use AI as a partner, not just a tool"
                ]
            }
        }

        filename = f"./prompts/collaboration_prompt_system_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(prompt_system, f, indent=2)

        print(f"💾 Prompt system saved to: {filename}")
        return filename

    def test_prompt_effectiveness(self, prompt_type: str) -> Dict[str, Any]:
        """Test how effective a specialized prompt is"""

        test_query = f"Help me with a {prompt_type} task. I need guidance on best practices and potential pitfalls."

        full_prompt = self.create_base_collaboration_prompt() + "\n\n" + self.prompt_templates.get(prompt_type, "")

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": test_query}
                ]
            )

            if response and "message" in response:
                ai_response = response["message"].get("content", "")

                # Analyze response quality
                analysis = {
                    "response_length": len(ai_response),
                    "has_structure": any(indicator in ai_response.lower() for indicator in ["step", "first", "consider", "recommend"]),
                    "shows_expertise": any(term in ai_response.lower() for term in ["best practice", "pattern", "consider", "trade-off"]),
                    "asks_questions": "?" in ai_response,
                    "provides_examples": any(word in ai_response.lower() for word in ["example", "instance", "case"])
                }

                return {
                    "prompt_type": prompt_type,
                    "test_query": test_query,
                    "ai_response": ai_response[:500] + "..." if len(ai_response) > 500 else ai_response,
                    "quality_analysis": analysis,
                    "overall_score": sum(analysis.values()) / len(analysis)
                }

        except Exception as e:
            return {"error": str(e), "prompt_type": prompt_type}

def main():
    """Main execution function"""

    print("🚀 Initializing Human-AI Collaboration Prompt Engineering System...")

    engineer = CollaborationPromptEngineer()

    print("📝 Creating specialized prompts...")
    prompts = engineer.create_specialized_prompts()

    print("🔧 Generating example conversations...")
    all_examples = {}
    for prompt_type in prompts.keys():
        print(f"  Generating examples for {prompt_type}...")
        examples = engineer.generate_examples(prompt_type, num_examples=2)
        all_examples[prompt_type] = examples

    print("📊 Testing prompt effectiveness...")
    test_results = {}
    for prompt_type in prompts.keys():
        print(f"  Testing {prompt_type} prompt...")
        result = engineer.test_prompt_effectiveness(prompt_type)
        test_results[prompt_type] = result

    print("💾 Saving complete system...")
    system_file = engineer.save_prompt_system()

    # Save examples and test results
    examples_file = f"./examples/collaboration_examples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(examples_file, 'w') as f:
        json.dump(all_examples, f, indent=2)

    results_file = f"./examples/prompt_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    print("🎉 Human-AI Collaboration System Complete!")
    print(f"📁 System saved to: {system_file}")
    print(f"📚 Examples saved to: {examples_file}")
    print(f"📊 Test results saved to: {results_file}")
    print("\n🤝 Ready for effective Human-AI collaboration!")

if __name__ == "__main__":
    main()