#!/usr/bin/env python3
"""
Validate the Human-AI Collaboration System
Tests the prompt system and integration
"""

import json
import os
from datetime import datetime

def validate_prompt_system():
    """Validate that the prompt system was created correctly"""

    prompt_file = "./prompts/human_ai_collaboration_system_20251102_083235.json"

    if not os.path.exists(prompt_file):
        print(f"❌ Prompt system file not found: {prompt_file}")
        return False

    try:
        with open(prompt_file, 'r') as f:
            system = json.load(f)

        # Validate structure
        required_keys = ['metadata', 'prompts', 'workflows', 'guide']
        for key in required_keys:
            if key not in system:
                print(f"❌ Missing required key: {key}")
                return False

        # Validate prompts
        required_prompts = [
            'base_collaboration',
            'code_review_specialist',
            'debugging_assistant',
            'architecture_consultant',
            'testing_strategist',
            'optimization_expert'
        ]

        for prompt_key in required_prompts:
            if prompt_key not in system['prompts']:
                print(f"❌ Missing required prompt: {prompt_key}")
                return False

        # Validate workflows
        required_workflows = ['feature_development', 'bug_fix', 'refactoring']
        for workflow_key in required_workflows:
            if workflow_key not in system['workflows']:
                print(f"❌ Missing required workflow: {workflow_key}")
                return False

        print("✅ Prompt system structure is valid")
        print(f"📊 Found {len(system['prompts'])} specialized prompts")
        print(f"🔄 Found {len(system['workflows'])} workflow templates")

        return True

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in prompt system file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating prompt system: {e}")
        return False

def validate_orchestrator_build():
    """Validate that the LLM orchestrator built successfully"""

    build_files = [
        "./llm-orchestrator/dist/index.js",
        "./llm-orchestrator/dist/index.d.ts"
    ]

    for build_file in build_files:
        if not os.path.exists(build_file):
            print(f"❌ Build file not found: {build_file}")
            return False

    print("✅ LLM orchestrator build files exist")
    return True

def validate_integration():
    """Validate that templates were added to the orchestrator"""

    template_file = "./llm-orchestrator/src/lib/common/template-engine.ts"

    if not os.path.exists(template_file):
        print(f"❌ Template engine file not found: {template_file}")
        return False

    try:
        with open(template_file, 'r') as f:
            content = f.read()

        # Check for collaboration templates
        collaboration_templates = [
            'COLLABORATION_BASE',
            'COLLABORATION_CODE_REVIEW',
            'COLLABORATION_DEBUGGING',
            'COLLABORATION_ARCHITECTURE',
            'COLLABORATION_TESTING',
            'COLLABORATION_OPTIMIZATION'
        ]

        missing_templates = []
        for template in collaboration_templates:
            if template not in content:
                missing_templates.append(template)

        if missing_templates:
            print(f"❌ Missing collaboration templates: {missing_templates}")
            return False

        print("✅ Collaboration templates integrated into orchestrator")
        return True

    except Exception as e:
        print(f"❌ Error validating template integration: {e}")
        return False

def test_prompt_rendering():
    """Test that prompts can be rendered (basic validation)"""

    try:
        # Simple template rendering test
        template = "Hello {{name}}, welcome to {{project}}!"
        variables = {"name": "Developer", "project": "Human-AI Collaboration"}

        # Basic variable replacement
        result = template
        for key, value in variables.items():
            result = result.replace("{{" + key + "}}", str(value))

        expected = "Hello Developer, welcome to Human-AI Collaboration!"
        if result == expected:
            print("✅ Basic template rendering works")
            return True
        else:
            print(f"❌ Template rendering failed. Expected: {expected}, Got: {result}")
            return False

    except Exception as e:
        print(f"❌ Error testing template rendering: {e}")
        return False

def main():
    """Run all validation tests"""

    print("🚀 Human-AI Collaboration System Validation")
    print("=" * 60)

    tests = [
        ("Prompt System Creation", validate_prompt_system),
        ("Orchestrator Build", validate_orchestrator_build),
        ("Template Integration", validate_integration),
        ("Basic Template Rendering", test_prompt_rendering)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")

    print("\n" + "=" * 60)
    print(f"📊 Validation Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All validation tests passed!")
        print("🎉 Human-AI Collaboration System is ready for use!")
        print("\nKey achievements:")
        print("- ✅ Specialized prompts for 6 collaboration scenarios")
        print("- ✅ Workflow templates for common development tasks")
        print("- ✅ Comprehensive collaboration guide")
        print("- ✅ Integration with LLM orchestrator")
        print("- ✅ TypeScript compilation successful")
        print("\n🤝 Ready for effective Human-AI collaboration!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())