"""Live app discovery, stable identity, fade, particle draining, and telemetry coverage."""
import math
import unittest

from lumo.process_sources import ProcessSources
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

    def test_rank_changes_preserve_existing_slots_and_colors(self):
        model=ProcessSources()
        initial=[sample(i,f"app-{i}",float(i)) for i in range(1,8)]
        model.update(initial,now=0.)
        before={s.name:(s.slot,s.color) for s in model.sources}
        changed=[sample(p.pid,p.name,1000. if p.pid==1 else 0.) for p in initial]
        model.update(changed,now=1.)
        self.assertEqual(before,{s.name:(s.slot,s.color) for s in model.sources})

    def test_new_app_forms_after_two_sensor_polls_and_does_not_move_existing_apps(self):
        model=ProcessSources()
        initial=[sample(i,f"app-{i}") for i in range(1,6)]
        model.update(initial,now=0.,flow_time=0.,sampled_at=1.)
        before={s.name:(s.slot,s.color) for s in model.sources}
        opened=initial+[sample(20,"new-app",30.)]
        model.update(opened,now=1.,flow_time=1.,sampled_at=2.)
        for frame in range(20):
            model.update(opened,now=1.+frame/100.,flow_time=1.,sampled_at=2.)
        self.assertNotIn("new-app",{s.name for s in model.sources})
        model.update(opened,now=2.,flow_time=2.,sampled_at=3.)
        source=next(s for s in model.sources if s.name=="new-app")
        self.assertEqual(source.opacity,0.)
        self.assertEqual(source.born_flow,2.)
        model.update(opened,now=3.5,flow_time=3.5,sampled_at=3.)
        self.assertAlmostEqual(source.opacity,.5)
        for s in model.sources:
            if s.name in before:
                self.assertEqual(before[s.name],(s.slot,s.color))
        self.assertNotIn(20,next(s for s in model.sources if s.name=="other processes").pids)

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

    def test_transient_process_does_not_get_a_cloud(self):
        model=ProcessSources()
        initial=[sample(1,"editor")]
        model.update(initial,now=0.)
        model.update(initial+[sample(2,"temporary")],now=1.)
        model.update(initial,now=2.)
        self.assertNotIn("temporary",{s.name for s in model.sources})

    def test_kernel_workers_remain_in_background(self):
        model=ProcessSources()
        initial=[sample(1,"editor")]
        model.update(initial,now=0.)
        with_worker=initial+[ProcessReading(2,"kworker/test",90.,0.)]
        model.update(with_worker,now=1.)
        model.update(with_worker,now=2.)
        self.assertNotIn("kworker/test",{s.name for s in model.sources})
        self.assertIn(2,next(s for s in model.sources if s.name=="other processes").pids)

    def test_capacity_overflow_and_slot_reuse_preserve_other_apps(self):
        model=ProcessSources(max_apps=2)
        initial=[sample(1,"first",10.),sample(2,"second",5.)]
        model.update(initial,now=0.,flow_time=0.)
        second=next(s for s in model.sources if s.name=="second")
        identity=second.slot,second.color
        opened=initial+[sample(3,"third")]
        model.update(opened,now=1.,flow_time=1.)
        model.update(opened,now=2.,flow_time=2.)
        self.assertIn(3,next(s for s in model.sources if s.name=="other processes").pids)
        remaining=opened[1:]
        model.update(remaining,now=3.,flow_time=3.)
        model.update(remaining,now=6.1,flow_time=6.1)
        third=next(s for s in model.sources if s.name=="third")
        self.assertEqual(third.slot,0)
        self.assertEqual((second.slot,second.color),identity)
        self.assertTrue(any(s.name=="first" and s.opacity==0. for s in model.sources))

    def test_reopen_keeps_color_and_gates_new_births(self):
        model=ProcessSources()
        model.update([sample(1,"editor")],now=0.,flow_time=0.)
        color=next(s for s in model.sources if s.name=="editor").color
        model.update([],now=1.,flow_time=1.)
        model.update([sample(2,"editor")],now=2.,flow_time=2.)
        model.update([sample(2,"editor")],now=3.,flow_time=3.)
        active=next(s for s in model.sources if s.name=="editor" and s.closed_at is None)
        self.assertEqual(active.color,color)
        self.assertEqual(active.born_flow,3.)
        self.assertEqual(active.pids,(2,))

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


if __name__=="__main__":
    unittest.main()
