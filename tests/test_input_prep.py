# -*- coding: utf-8 -*-
"""输入预处理: 清洗 + 复合问题拆分"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.input_prep import clean_user_input, split_compound_question


class TestCleanUserInput(unittest.TestCase):
    def test_strip_replacement_char(self):
        self.assertEqual(clean_user_input("上次差\uFFFD查的论文"), "上次差查的论文")

    def test_fullwidth_to_halfwidth(self):
        self.assertEqual(clean_user_input("ＡＢＣ１２３"), "ABC123")

    def test_fullwidth_punct(self):
        self.assertEqual(clean_user_input("什么是ＧＡＮ？"), "什么是GAN?")

    def test_collapse_repeated_punct(self):
        self.assertEqual(clean_user_input("真的吗？？？"), "真的吗？")
        self.assertEqual(clean_user_input("太棒了！！！"), "太棒了！")

    def test_zero_width_removed(self):
        self.assertEqual(clean_user_input("GAN\u200b是什么"), "GAN是什么")

    def test_whitespace_normalized(self):
        self.assertEqual(clean_user_input("  查一下  GAN  \n 的资料  "), "查一下 GAN 的资料")

    def test_empty(self):
        self.assertEqual(clean_user_input(""), "")
        self.assertEqual(clean_user_input(None), "")


class TestSplitCompoundQuestion(unittest.TestCase):
    def test_multi_question(self):
        primary, extras = split_compound_question("什么是GAN？顺便查一下贝叶斯论文")
        self.assertEqual(primary, "什么是GAN")
        self.assertEqual(extras, ["顺便查一下贝叶斯论文"])

    def test_connector_marker(self):
        primary, extras = split_compound_question("讲讲ESN顺便说说它的缺点")
        self.assertEqual(primary, "讲讲ESN")
        self.assertIn("顺便说说它的缺点", extras)

    def test_another_marker(self):
        primary, extras = split_compound_question("总结刘月的论文，另外对比一下MCMC")
        self.assertEqual(primary, "总结刘月的论文")
        self.assertIn("另外对比一下MCMC", extras)

    def test_no_split(self):
        primary, extras = split_compound_question("什么是GAN")
        self.assertEqual(primary, "什么是GAN")
        self.assertEqual(extras, [])

    def test_short_no_split(self):
        primary, extras = split_compound_question("顺便看看")  # 主段太短不拆
        self.assertEqual(primary, "顺便看看")
        self.assertEqual(extras, [])

    def test_three_questions_cap(self):
        primary, extras = split_compound_question("GAN是什么？怎么训练？有哪些变体？")
        self.assertEqual(primary, "GAN是什么")
        self.assertEqual(len(extras), 2)  # 最多 2 个附加

    def test_fragment_questions_not_split(self):
        # 单字符碎问句(好？不好？)不拆
        primary, extras = split_compound_question("好？不好？")
        self.assertEqual(primary, "好？不好？")
        self.assertEqual(extras, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
