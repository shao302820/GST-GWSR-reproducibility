import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src/gwsr/exact"))
import unittest
import copy
import numpy as np
import archive_engine as engine


class Program:
    def __init__(self,token,score):
        self.program=[float(token)]
        self.fitness_=float(score)


class ArchiveTests(unittest.TestCase):
    def test_duplicate_structure_keeps_best(self):
        a=engine.EliteArchive(3);a.update([Program(1,4),Program(1,2)],1)
        self.assertEqual(len(a.pool()),1);self.assertEqual(a.snapshot()['best_score'],2)

    def test_old_elite_survives_later_worse_generation(self):
        a=engine.EliteArchive(3);a.update([Program(1,1),Program(2,2)],1)
        a.update([Program(3,10),Program(4,20)],2)
        self.assertEqual([p.fitness_ for p in a.pool()],[1,2,10])

    def test_capacity_and_order(self):
        a=engine.EliteArchive(2);a.update([Program(i,5-i) for i in range(5)],1)
        self.assertEqual([p.fitness_ for p in a.pool()],[1,2])

    def test_constants_not_rounded(self):
        self.assertNotEqual(engine.signature(Program(.12341,1)),engine.signature(Program(.12349,1)))

    def test_copies_do_not_follow_later_mutations(self):
        p=Program(1,1);a=engine.EliteArchive();a.update([p],1)
        p.program[0]=7.;p.fitness_=100.
        self.assertEqual(a.pool()[0].program,[1.]);self.assertEqual(a.pool()[0].fitness_,1.)

    def test_nonfinite_is_excluded(self):
        a=engine.EliteArchive();a.update([Program(1,np.inf),Program(2,.2)],1)
        self.assertEqual(len(a.pool()),1)

    def test_repeat_fill_is_deterministic(self):
        selected,info=engine.take800([Program(1,1),Program(2,2)])
        self.assertEqual(len(selected),800);self.assertEqual(info['incoming_duplicate_fill'],798)
        self.assertEqual(selected[:4],[selected[0],selected[1],selected[0],selected[1]])

    def test_no_execute_needed(self):
        # Fake candidates have no execute method: archive management must still work.
        a=engine.EliteArchive();a.update([Program(1,.1)],1)
        self.assertEqual(a.snapshot()['best_score'],.1)

    def test_exact_800_needs_no_fill(self):
        a=engine.EliteArchive();a.update([Program(i,i) for i in range(1000)],1)
        pool,info=engine.take800(a.pool())
        self.assertEqual(len(pool),800);self.assertEqual(info['incoming_duplicate_fill'],0)
        self.assertEqual(pool[-1].fitness_,799.)


if __name__=='__main__':unittest.main(verbosity=2)
