def cleaned(text):
    return text.replace("_", "-")


def joined_line(row):
    return " & ".join(row) + " \\\\"


def tabular(headers, rows):
    cells = [[cleaned(cell) for cell in row] for row in [headers] + rows]
    lines = [
        "\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}",
        "\\toprule",
        joined_line(cells[0]),
        "\\midrule",
    ]

    for row in cells[1:]:
        lines.append(joined_line(row))

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)
