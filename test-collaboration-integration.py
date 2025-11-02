#!/usr/bin/env python3
"""
Test the Human-AI Collaboration System
Validates the integration with Ollama granite4:3b model
"""

import json
import sys
import os

# Add the built llm-orchestrator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'llm-orchestrator', 'dist'))

try:
    from index import codeReviewCollaboration
    print("✅ Successfully imported collaboration functions")
except ImportError as e:
    print(f"❌ Failed to import collaboration functions: {e}")
    sys.exit(1)

async def test_code_review():
    """Test the code review collaboration function"""

    test_code = """
function calculateTotal(items) {
    let total = 0;
    for (let i = 0; i < items.length; i++) {
        total += items[i].price * items[i].quantity;
    }
    return total;
}
"""

    print("🧪 Testing code review collaboration...")
    print("Code to review:")
    print(test_code.strip())

    try:
        result = await codeReviewCollaboration(
            code=test_code,
            language="javascript",
            context="This is a simple e-commerce calculation function"
        )

        print("✅ Code review completed successfully!")
        print(f"Model: {result['result']['model']}")
        print(f"Task Type: {result['result']['taskType']}")
        print("Response preview:")
        print(result['result']['content'][:300] + "..." if len(result['result']['content']) > 300 else result['result']['content'])

        return True

    except Exception as e:
        print(f"❌ Code review test failed: {e}")
        return False

async def test_debugging():
    """Test the debugging collaboration function"""

    from src.lib.actions.human_ai_collaboration import debuggingCollaboration

    issue_description = "My React component is re-rendering too frequently and causing performance issues"

    print("\n🧪 Testing debugging collaboration...")
    print(f"Issue: {issue_description}")

    try:
        result = await debuggingCollaboration(
            issue_description,
            context={
                "error_messages": "No specific errors, just slow performance",
                "code": "React component with useEffect and state updates"
            }
        )

        print("✅ Debugging assistance completed successfully!")
        print("Response preview:")
        print(result['result']['content'][:300] + "..." if len(result['result']['content']) > 300 else result['result']['content'])

        return True

    except Exception as e:
        print(f"❌ Debugging test failed: {e}")
        return False

async def main():
    """Run all validation tests"""

    print("🚀 Starting Human-AI Collaboration System Validation")
    print("=" * 60)

    # Test 1: Code Review
    test1_passed = await test_code_review()

    # Test 2: Debugging
    test2_passed = await test_debugging()

    print("\n" + "=" * 60)
    print("📊 Validation Results:")

    if test1_passed and test2_passed:
        print("✅ All tests passed! Human-AI collaboration system is working correctly.")
        print("🤝 The specialized prompts are successfully integrated with Ollama granite4:3b")
        return 0
    else:
        print("❌ Some tests failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)