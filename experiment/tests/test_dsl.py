import unittest

from experiment.dsl import (
    GRID_SIZE,
    Program,
    call,
    execute,
    grid_from_rows,
    primitive,
    program_size,
    program_to_string,
    render_grid,
)


class DslTests(unittest.TestCase):
    def test_primitive_grids_are_in_bounds(self):
        for name in [
            "blank",
            "line_horizontal",
            "line_vertical",
            "diagonal",
            "square",
            "triangle",
        ]:
            grid = execute(primitive(name))
            self.assertIsInstance(grid, frozenset)
            self.assertTrue(all(0 <= row < GRID_SIZE for row, _ in grid))
            self.assertTrue(all(0 <= col < GRID_SIZE for _, col in grid))

    def test_basic_primitive_shapes(self):
        horizontal = execute(primitive("line_horizontal"))
        vertical = execute(primitive("line_vertical"))
        diagonal = execute(primitive("diagonal"))
        square = execute(primitive("square"))
        triangle = execute(primitive("triangle"))

        self.assertEqual(horizontal, frozenset((5, col) for col in range(10)))
        self.assertEqual(vertical, frozenset((row, 5) for row in range(10)))
        self.assertEqual(diagonal, frozenset((i, i) for i in range(10)))
        self.assertIn((0, 0), square)
        self.assertIn((9, 9), square)
        self.assertNotIn((5, 5), square)
        self.assertIn((9, 0), triangle)
        self.assertIn((9, 9), triangle)
        self.assertNotIn((0, 9), triangle)

    def test_binary_ops(self):
        horizontal = primitive("line_horizontal")
        vertical = primitive("line_vertical")

        union = execute(call("add", horizontal, vertical))
        overlap = execute(call("overlap", horizontal, vertical))
        alias_overlap = execute(call("intersect", horizontal, vertical))
        subtract = execute(call("subtract", horizontal, vertical))

        self.assertEqual(union, execute(horizontal) | execute(vertical))
        self.assertEqual(overlap, frozenset({(5, 5)}))
        self.assertEqual(alias_overlap, overlap)
        self.assertNotIn((5, 5), subtract)
        self.assertIn((5, 0), subtract)

    def test_unary_ops_are_involutions(self):
        program = call("add", primitive("line_horizontal"), primitive("triangle"))

        for op in [
            "invert",
            "reflect_horizontal",
            "reflect_vertical",
            "reflect_diag",
        ]:
            self.assertEqual(execute(call(op, call(op, program))), execute(program))

    def test_render_and_parse_grid(self):
        rows = [
            "#.........",
            ".#........",
            "..#.......",
            "...#......",
            "....#.....",
            ".....#....",
            "......#...",
            ".......#..",
            "........#.",
            ".........#",
        ]
        grid = grid_from_rows(rows)
        self.assertEqual(render_grid(grid), "\n".join(rows))

    def test_program_string_and_size(self):
        program = call("intersect", primitive("line_horizontal"), primitive("square"))

        self.assertEqual(program_to_string(program), "overlap(line_horizontal,square)")
        self.assertEqual(program_size(program), 3)

    def test_bad_programs_raise(self):
        with self.assertRaises(ValueError):
            execute(primitive("not_real"))
        with self.assertRaises(ValueError):
            execute(call("add", primitive("blank")))
        with self.assertRaises(ValueError):
            execute(call("invert", primitive("blank"), primitive("blank")))

    def test_helpers_behave_like_primitives(self):
        helper_grid = execute(call("add", primitive("line_horizontal"), primitive("diagonal")))

        self.assertEqual(execute(Program("H1"), helpers={"H1": helper_grid}), helper_grid)

    def test_invalid_grid_rows_raise(self):
        with self.assertRaises(ValueError):
            grid_from_rows(["." * GRID_SIZE])
        with self.assertRaises(ValueError):
            grid_from_rows(["." * GRID_SIZE for _ in range(GRID_SIZE - 1)] + ["x" * GRID_SIZE])


if __name__ == "__main__":
    unittest.main()
