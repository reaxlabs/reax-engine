# SPDX-License-Identifier: MIT
import math
import unittest
from probabilities import mix


class ProbabilityContract(unittest.TestCase):
    def setUp(self):
        self.r = {'questions': {'q': {'type': 'choice', 'criteria': {'a': 'A', 'b': 'B', 'c': 'C'}}}}
        self.g = {'answers': {'q': {'type': 'choice', 'probabilities': {'a': .6, 'b': .3, 'c': .1}}}}
        self.q = {'answers': {'q': {'type': 'choice', 'probabilities': {'a': .2, 'b': .3, 'c': .5}}}}

    def test_equal_probability_average(self):
        answer = mix(self.r, self.g, self.q)['answers']['q']
        self.assertEqual(set(answer['probabilities']), {'a','b','c'})
        for key, expected in {'a': .4,'b': .3,'c': .3}.items():
            self.assertAlmostEqual(answer['probabilities'][key], expected)
        self.assertEqual(answer['choice'], 'a')
        self.assertAlmostEqual(math.fsum(answer['probabilities'].values()), 1, places=15)

    def test_missing_option_rejected(self):
        del self.q['answers']['q']['probabilities']['c']
        with self.assertRaises(ValueError): mix(self.r,self.g,self.q)

    def test_invalid_native_probabilities_rejected(self):
        for value in (float('nan'),float('inf'),-1,2):
            self.q['answers']['q']['probabilities']['c']=value
            with self.assertRaises(ValueError): mix(self.r,self.g,self.q)

    def test_unnormalized_rejected(self):
        self.q['answers']['q']['probabilities']={'a':.1,'b':.1,'c':.1}
        with self.assertRaises(ValueError): mix(self.r,self.g,self.q)

    def test_question_mismatch_rejected(self):
        self.q['answers']={}
        with self.assertRaises(ValueError): mix(self.r,self.g,self.q)

    def test_noul_and_score(self):
        r={'questions':{'n':{'type':'noul'},'s':{'type':'score','criteria':['bad','good']}}}
        g={'answers':{'n':{'type':'noul','noul':.2},'s':{'type':'score','probabilities':{'0':.3,'1':.7}}}}
        q={'answers':{'n':{'type':'noul','noul':.8},'s':{'type':'score','probabilities':{'0':.7,'1':.3}}}}
        result=mix(r,g,q)['answers']
        self.assertEqual(result['n']['noul'],.5)
        self.assertEqual(result['s']['score'],.5)
        self.assertEqual(result['s']['probabilities'],{'0':.5,'1':.5})

if __name__=='__main__': unittest.main()
