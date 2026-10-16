"""Live app discovery, stable identity, fade, particle draining, and telemetry coverage."""
import math
import unittest

from lumo.process_sources import ProcessSources, SOURCE_COLORS
from lumo.telemetry import ProcessReading


def sample(pid,name,cpu=0.):
    return ProcessReading(pid,name,cpu,2**24)


class ProcessSourceTests(unittest.TestCase):
    def test_program_groups_cover_all_pids_once_and_sum_readings(self):
        readings=[ProcessReading(i,"worker" if i<3 else f"app-{i}",float(i),1024.*i) for i in range(1,10)]
        sources=ProcessSources().update(readings+[readings[0]],now=0.,flow_time=0.)
        pids=[pid for source in sources for pid in source.pids]
        self.assertCountEqual(pids,list(range(1,10)))
        self.assertEqual(len(pids),len(set(pids)))
        self.assertEqual(sum(s.cpu for s in sources),45.)
        self.assertEqual(sum(s.memory for s in sources),45.*1024.)

    def test_close_fades_cloud_but_preserves_particle_emission_until_drained(self):
        model=ProcessSources()
        model.update([sample(1,"editor",60.)],now=0.,flow_time=0.)
        source=next(s for s in model.sources if s.name=="editor")
        strength=source.render_emission
        model.update([],now=1.,flow_time=1.)
        model.update([],now=2.5,flow_time=2.5)
        self.assertAlmostEqual(source.opacity,.5)
        self.assertEqual(source.stopped_flow,1.)
        self.assertEqual(source.render_emission,strength)
        model.update([],now=5.,flow_time=5.)
        self.assertEqual(source.opacity,0.)
        self.assertIn(source,model.sources)
        model.update([],now=36.,flow_time=36.)
        self.assertNotIn(source,model.sources)

    def test_source_clocks_follow_own_load_without_jumping_or_losing_draining_particles(self):
        model=ProcessSources()
        readings=[sample(1,"busy",100.),sample(2,"idle",0.)]
        model.update(readings,now=0.,flow_time=0.,source_speeds={"busy":1.1,"idle":.15})
        for step in range(1,61):
            model.update(readings,now=step/10.,flow_time=step/10.,
                         source_speeds={"busy":1.1,"idle":.15})
        busy=next(s for s in model.sources if s.name=="busy")
        idle=next(s for s in model.sources if s.name=="idle")
        self.assertGreater(busy.flow_clock,idle.flow_clock*4)
        previous=busy.flow_clock
        model.update(readings,now=6.,flow_time=100.,source_speeds={"busy":.15,"idle":.15})
        self.assertEqual(busy.flow_clock,previous)
        model.update([readings[1]],now=6.1,flow_time=101.,source_speeds={"idle":.15})
        self.assertEqual(busy.stopped_flow,busy.flow_clock)
        self.assertIn(busy,model.sources)
        closing_speed=busy.flow_speed
        model.update([readings[1]],now=9.,flow_time=104.,source_speeds={"idle":.15})
        self.assertLess(busy.flow_speed,closing_speed*.4)
        self.assertIn(busy,model.sources)

    def test_cpu_increases_emission_and_invalid_counters_stay_finite(self):
        model=ProcessSources()
        idle=model.update([sample(1,"worker")],now=0.)[0].emission
        low=model.update([sample(1,"worker",20.)],now=1.)[0].emission
        high=model.update([sample(1,"worker",100.)],now=2.)[0].emission
        self.assertLess(idle,low)
        self.assertLess(low,high)
        source=model.update([ProcessReading(1,"worker",math.nan,math.inf)],now=3.)[0]
        self.assertTrue(math.isfinite(source.emission))
        self.assertEqual(source.cpu,0.)
        self.assertEqual(source.memory,0.)

    def test_sustained_internal_reversal_and_admission(self):
        model=ProcessSources(max_apps=2)
        model.update([sample(1,'a',20),sample(2,'b',10)],now=0, sampled_at=0)
        changed=[sample(1,'a',20),sample(2,'b',40),sample(3,'c',60)]
        def order():
            return [s.name for s in sorted(model._live.values(),key=lambda s:s.slot)]
        for t in (1,2,3):
            model.update(changed,now=t,sampled_at=t)
            self.assertEqual(order(),['a','b'])
        model.update(changed,now=4,sampled_at=4)
        self.assertEqual(order(),['c','b'])
        self.assertEqual(len({p for s in model.sources for p in s.pids}),3)
        for source in model.sources:
            self.assertEqual(source.color,SOURCE_COLORS[source.slot])
            if source.closed_at is not None:
                self.assertEqual(source.pids,())
                self.assertEqual(source.emission,0)

    def test_broken_margin_repeated_and_delayed_samples_reset(self):
        for mode in ('broken','repeated','delayed'):
            with self.subTest(mode=mode):
                model=ProcessSources(max_apps=1)
                initial=[sample(1,'a',10),sample(2,'b',0)]
                changed=[sample(1,'a',10),sample(2,'b',14)]
                model.update(initial,now=0,sampled_at=0)
                model.update(changed,now=1,sampled_at=1)
                if mode=='broken':
                    model.update([sample(1,'a',10),sample(2,'b',12)],now=2,sampled_at=2)
                    model.update(changed,now=3,sampled_at=3)
                    model.update(changed,now=4,sampled_at=4)
                elif mode=='repeated':
                    model.update(changed,now=4,sampled_at=1)
                else:
                    model.update(changed,now=5,sampled_at=5)
                self.assertIn('a',model._live)

    def test_ties_full_names_readiness_and_cpu_normalization(self):
        model=ProcessSources(max_apps=1)
        model.update([sample(1,'long name alpha',10),sample(2,'long name beta',10)],now=0)
        self.assertIn('long name alpha',model._live)
        for t in range(1,6):
            model.update([sample(1,'long name alpha',10),sample(2,'long name beta',14)],now=t,logical_cpus=2)
        self.assertIn('long name alpha',model._live)

    def test_disappearance_reopen_and_repeated_swaps_are_bounded(self):
        model=ProcessSources(max_apps=1)
        model.update([sample(1,'a',10)],now=0)
        old=model._live['a']
        model.update([],now=1)
        model.update([sample(2,'a',10)],now=2)
        self.assertIsNot(old,model._live['a'])
        self.assertEqual(old.pids,())
        for t in range(3,100):
            model.update([sample(t,'a' if t%2 else 'b',10)],now=t,flow_time=0)
        self.assertLessEqual(len(model._retiring),12)

    def test_unready_and_stale_samples_do_not_supply_rank_evidence(self):
        model=ProcessSources(max_apps=1)
        model.update([sample(1,'a',10),sample(2,'b',0)],now=0,sampled_at=0)
        changed=[sample(1,'a',10),sample(2,'b',20)]
        model.update(changed,now=1,sampled_at=1)
        model.update([sample(1,'a',10),ProcessReading(2,'b',20,100,False)],now=2,sampled_at=2)
        for t in (3,4,5):
            model.update(changed,now=t,sampled_at=t)
            self.assertIn('a',model._live)
        model.update(changed,now=6,sampled_at=6)
        self.assertIn('b',model._live)
        owners=[pid for source in model.live_sources for pid in source.pids]
        self.assertCountEqual(owners,[1,2])
        self.assertTrue(all(not s.pids for s in model.retiring_sources))
        model.update([sample(1,'a',100)],now=7,sampled_at=4)
        self.assertIn('b',model._live)

    def test_named_policy_and_reduced_capacity(self):
        model=ProcessSources(names=['b','a'],max_apps=1)
        model.update([sample(1,'a',99),sample(2,'b',0)],now=0)
        self.assertEqual(list(model._live),['b'])
        self.assertEqual(model._background.pids,(1,))


if __name__=="__main__":
    unittest.main()
