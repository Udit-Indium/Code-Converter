from __future__ import annotations
import ast

def first_line(node: ast.stmt) -> int:
    """The real first line of `node`, counting decorators.

    `node.lineno` on a decorated function points at the `def`, not at the `@`,
    so slicing from it would drop the decorators — and a dropped decorator is a
    silent behaviour change.
    """
    lines = [node.lineno]
    for decorator in getattr(node, "decorator_list", []):
        lines.append(decorator.lineno)
    return min(lines)


class SourceMap:
    """Maps AST statements back to their original text.

    Args:
        source: the complete source file, exactly as read.
    """

    def __init__(self, source: str) -> None:
        self._lines = source.splitlines()

    def segment(self, node: ast.stmt) -> str:
        """Return `node`'s own lines, verbatim.

        Whole lines are taken rather than the precise column span, so a trailing
        comment comes along with the statement it annotates.
        """
        start = first_line(node)
        end = node.end_lineno or start
        return "\n".join(self._lines[start - 1 : end])

    def leading_comments(self, node: ast.stmt, floor: int) -> str:
        """Return the comment block sitting immediately above `node`.

        Walks upward collecting full-line comments, stopping at the first line
        that is neither a comment nor blank. Blank lines are crossed but not
        kept at the outer edge, so::

            x = 1
                            <- blank, crossed and dropped
            # about y       <- kept
            y = 2

        gives `y` the comment and nothing else.

        Args:
            node: the statement claiming the comments.
            floor: last line already consumed by the previous statement.
                Nothing at or below this is available, which stops a statement
                from stealing text that belongs to its predecessor.
        """
        start = first_line(node)
        collected: list[str] = []
        cursor = start - 1 

        while cursor > floor:
            line = self._lines[cursor - 1]
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                collected.append(line)
                cursor -= 1
                continue
            break

        collected.reverse()
        while collected and not collected[0].strip():
            collected.pop(0)
        while collected and not collected[-1].strip():
            collected.pop()
        return "\n".join(collected)

    def statement_source(self, node: ast.stmt, floor: int) -> str:
        """Return `node` with its leading comment block attached."""
        comments = self.leading_comments(node, floor)
        body = self.segment(node)
        return f"{comments}\n{body}" if comments else body
