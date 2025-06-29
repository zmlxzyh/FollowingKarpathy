from graphviz import Digraph


def trace(root):
    """
    从最终输出节点 root 出发，递归回溯整个计算图，
    收集所有 Value 节点和它们之间的依赖关系（边）。
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)  # 加入当前节点
            for child in v._prev:  # 遍历它的前驱节点
                edges.add((child, v))  # 添加边：child -> v
                build(child)  # 递归继续向上回溯

    build(root)
    return nodes, edges  # 返回所有节点和边


def draw_dot(root):
    """
    画出从 root 出发的整个计算图（使用 graphviz），
    节点显示其 data 值，若是计算节点还显示操作符。
    """
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # 从左向右画图

    nodes, edges = trace(root)  # 追踪图中的所有节点与边

    for n in nodes:
        uid = str(id(n))  # 使用对象唯一 ID 作为图中节点名称，防止重复

        # 创建一个表示变量数值的矩形节点
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )

        # 如果这个值是由某个操作生成的，插入一个操作节点
        if n._op:
            dot.node(name=uid + n._op, label=n._op)  # 操作符节点（如 +, *, relu）
            dot.edge(uid + n._op, uid)  # 操作 -> 输出值

    for n1, n2 in edges:
        # 将边连接到操作符节点，而不是值节点本身
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
