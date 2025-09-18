from manim import *

class GenScene(Scene):
    def construct(self):
        axes = Axes(
            x_range=[0, 40, 5],
            y_range=[0, 120, 20],
            x_length=7,
            y_length=5,
            axis_config={"include_tip": True},
            x_axis_config={"numbers_to_include": [0, 16, 40], "label_direction": DOWN},
            y_axis_config={"numbers_to_include": [0, 68, 120], "label_direction": LEFT},
        ).to_edge(DOWN)

        x_label = axes.get_x_axis_label("Q")
        y_label = axes.get_y_axis_label("P")

        demand_graph = axes.plot(lambda q: 100 - 2 * q, x_range=[0, 50], color=BLUE)
        demand_label = MathTex("P=100-2Q").next_to(demand_graph.get_end(), UP)

        supply_graph = axes.plot(lambda q: 20 + 3 * q, x_range=[0, 30], color=RED)
        supply_label = MathTex("P=20+3Q").next_to(supply_graph.get_end(), DOWN)

        equilibrium_q = 16
        equilibrium_p = 68
        eq_point = Dot(axes.c2p(equilibrium_q, equilibrium_p), color=GREEN)
        eq_label = MathTex("Q^*=16,\\ P^*=68").next_to(eq_point, UP + RIGHT, buff=0.2)

        v_line = DashedLine(axes.c2p(equilibrium_q, 0), axes.c2p(equilibrium_q, equilibrium_p), color=GREEN)
        h_line = DashedLine(axes.c2p(0, equilibrium_p), axes.c2p(equilibrium_q, equilibrium_p), color=GREEN)

        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(demand_graph), Write(demand_label))
        self.play(Create(supply_graph), Write(supply_label))
        self.play(Create(eq_point), Write(eq_label))
        self.play(Create(v_line), Create(h_line))