#!/usr/bin/env python3
"""
Human-AI Collaboration Prompt System
Creates specialized prompts for effective collaboration
"""

import json
import os
from datetime import datetime

class CollaborationPromptSystem:
    def __init__(self):
        self.prompts = {}
        os.makedirs("./prompts", exist_ok=True)

    def create_base_system(self):
        """Create the core collaboration system"""

        self.prompts = {
            "base_collaboration": """You are an expert Human-AI collaboration assistant with deep knowledge of software development, system design, and collaborative workflows.

CORE PRINCIPLES:
1. **Clarity First**: Always explain your reasoning and suggestions clearly
2. **Context Awareness**: Consider the user's goals, constraints, and existing codebase
3. **Step-by-Step Guidance**: Break complex tasks into manageable steps
4. **Best Practices**: Follow industry standards and proven patterns
5. **Ethical AI**: Be transparent about limitations and uncertainties
6. **Collaborative Mindset**: Work as a true partner, not just a code generator

COLLABORATION APPROACH:
- Ask clarifying questions when requirements are ambiguous
- Provide multiple approaches when appropriate
- Explain trade-offs and considerations
- Suggest testing and validation strategies
- Help with implementation details
- Review and improve upon ideas

Remember: You're not just writing code - you're helping build better software through collaboration.""",

            "code_review_specialist": """
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

        return self.prompts

    def create_workflow_templates(self):
        """Create workflow templates for common tasks"""

        workflows = {
            "feature_development": {
                "steps": [
                    "Requirements Analysis - Clarify goals and constraints",
                    "Design Planning - Architecture and approach",
                    "Implementation - Write and test code",
                    "Code Review - Quality assurance",
                    "Testing - Validate functionality",
                    "Documentation - Update guides and docs",
                    "Deployment - Release to production"
                ],
                "checkpoints": [
                    "Requirements signed off",
                    "Design reviewed and approved",
                    "Core functionality working",
                    "All tests passing",
                    "Documentation updated"
                ]
            },

            "bug_fix": {
                "steps": [
                    "Reproduce Issue - Confirm the problem",
                    "Investigate Root Cause - Find the source",
                    "Design Fix - Plan the solution",
                    "Implement Fix - Apply the changes",
                    "Test Fix - Verify the solution",
                    "Regression Testing - Ensure no new issues"
                ],
                "checkpoints": [
                    "Issue reproduced reliably",
                    "Root cause identified",
                    "Fix implemented and tested",
                    "No regressions introduced"
                ]
            },

            "refactoring": {
                "steps": [
                    "Analyze Impact - Understand dependencies",
                    "Plan Refactoring - Design the changes",
                    "Implement Changes - Apply incrementally",
                    "Test Continuously - Validate after each change",
                    "Performance Check - Ensure no degradation",
                    "Code Review - Quality assurance"
                ],
                "checkpoints": [
                    "Impact analysis complete",
                    "Test coverage verified",
                    "All changes tested",
                    "Performance maintained"
                ]
            }
        }

        return workflows

    def create_collaboration_guide(self):
        """Create a comprehensive collaboration guide"""

        guide = {
            "introduction": "This guide helps you collaborate effectively with AI assistants on software development tasks.",

            "collaboration_principles": [
                "Be specific about your goals and constraints",
                "Provide context about your codebase and technology stack",
                "Break complex tasks into smaller, manageable steps",
                "Ask for clarification when suggestions are unclear",
                "Review and test AI-generated code before using in production",
                "Use AI for learning and understanding, not just code generation",
                "Maintain clear communication about expectations",
                "Celebrate collaborative successes and learn from challenges"
            ],

            "effective_prompts": {
                "structure": "Start with context, state the task, specify constraints, request format",
                "clarity": "Use clear, specific language. Avoid ambiguity.",
                "context": "Include relevant background, existing code, and requirements",
                "examples": "Show examples of desired input/output when possible",
                "iteration": "Use follow-up prompts to refine and improve"
            },

            "common_patterns": {
                "code_generation": "Specify language, framework, patterns, and testing requirements",
                "code_review": "Provide the code, context, and specific areas of concern",
                "debugging": "Include error messages, reproduction steps, and relevant code",
                "architecture": "Describe requirements, constraints, scale, and success criteria",
                "optimization": "Share current performance metrics and specific bottlenecks",
                "documentation": "Specify audience, purpose, and key information to include"
            },

            "quality_assurance": [
                "Always review AI-generated code for correctness",
                "Test thoroughly, especially edge cases",
                "Consider security implications",
                "Verify performance requirements are met",
                "Ensure code follows your team's conventions",
                "Check for proper error handling"
            ]
        }

        return guide

    def save_system(self):
        """Save the complete collaboration system"""

        system = {
            "metadata": {
                "name": "Human-AI Collaboration System",
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "purpose": "Enhanced collaboration between humans and AI assistants"
            },
            "prompts": self.prompts,
            "workflows": self.create_workflow_templates(),
            "guide": self.create_collaboration_guide()
        }

        filename = f"./prompts/human_ai_collaboration_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(system, f, indent=2, ensure_ascii=False)

        print(f"💾 Collaboration system saved to: {filename}")
        return filename

def main():
    print("🚀 Creating Human-AI Collaboration System...")

    system = CollaborationPromptSystem()

    print("📝 Building prompt templates...")
    prompts = system.create_base_system()

    print("🔧 Creating workflow templates...")
    workflows = system.create_workflow_templates()

    print("📚 Building collaboration guide...")
    guide = system.create_collaboration_guide()

    print("💾 Saving complete system...")
    filename = system.save_system()

    print("🎉 Human-AI Collaboration System Complete!")
    print(f"📁 System saved to: {filename}")
    print("\n🤝 Ready for effective Human-AI collaboration!")
    print("\nKey components:")
    print("- Specialized prompts for different development scenarios")
    print("- Workflow templates for common tasks")
    print("- Comprehensive collaboration guide")
    print("- Best practices and quality assurance guidelines")

if __name__ == "__main__":
    main()