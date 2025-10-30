from evoprompt.utils.text import safe_format


def test_safe_format_preserves_braces_in_value():
    template = "Code snippet:\n{input}"
    code = "int main() { return arr[index]; }"

    formatted = safe_format(template, input=code)

    assert formatted == "Code snippet:\nint main() { return arr[index]; }"


def test_safe_format_handles_literal_braces_in_template():
    template = (
        "Respond with JSON object like {\"label\": \"vulnerable\"} if needed.\n"
        "Snippet:\n{input}"
    )
    code = "void f() { printf(\"hello\"); }"

    formatted = safe_format(template, input=code)

    expected = (
        "Respond with JSON object like {\"label\": \"vulnerable\"} if needed.\n"
        "Snippet:\nvoid f() { printf(\"hello\"); }"
    )
    assert formatted == expected
