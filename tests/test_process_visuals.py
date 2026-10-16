import math
import unittest

from lumo.process_sources import ProcessSources
from lumo.process_visuals import build_slots, SLOT_ANCHORS
from lumo.telemetry import ProcessReading, Snapshot


class VisualTests(unittest.TestCase):
    def fixture(self, gpu=None, total=1000):
        readings=[ProcessReading(i, 'LUMO_'+str(i),60-i*5,10*i) for i in range(1,8)]
        snap=Snapshot(cores=[0,0],ram_total=total,ram_used=600,gpu=gpu,
                      process_details=readings+[readings[0]])
        model=ProcessSources()
        model.update(snap.process_details,now=0,logical_cpus=2)
        return build_slots(model,snap)

    def test_resources_and_worker_accounting(self):
        slots=self.fixture(75)
        self.assertEqual(len(slots),6)
        self.assertAlmostEqual(slots[0].cpu_percent,27.5)
        self.assertEqual(slots[3].gpu_percent,75)
        self.assertEqual(slots[3].activity,.75)
        self.assertEqual(slots[4].total_ram_percent,60)
        self.assertAlmostEqual(slots[4].worker_ram_percent,28)
        self.assertAlmostEqual(slots[4].worker_ram_share,280/600*100)
        self.assertEqual(slots[4].activity,.6)
        self.assertEqual(slots[5].name,'Other programs')
        self.assertEqual(slots[0].anchor,SLOT_ANCHORS[0])

    def test_unavailable_gpu_falls_back_to_program_cpu(self):
        slots=self.fixture()
        self.assertIsNone(slots[3].gpu_percent)
        self.assertEqual(slots[3].activity,slots[3].cpu_percent/100)

    def test_invalid_totals_and_gpu(self):
        for total in (0,-1,math.nan,math.inf):
            slots=self.fixture(math.nan,total)
            self.assertIsNone(slots[4].total_ram_percent)
            self.assertIsNone(slots[4].worker_ram_percent)
            self.assertIsNone(slots[0].ram_percent)
            self.assertEqual(slots[4].activity,0)

    def test_empty_positions_have_no_metrics_or_streams(self):
        model=ProcessSources()
        snap=Snapshot(gpu=99,ram_used=90,ram_total=100,
                      process_details=[ProcessReading(1,'app',0,10,False)])
        model.update(snap.process_details,now=0)
        slots=build_slots(model,snap)
        self.assertIsNone(slots[0].cpu_percent)
        self.assertEqual(slots[0].ram_percent,10)
        for slot in slots[1:]:
            self.assertFalse(slot.occupied)
            self.assertEqual(slot.name,'—')
            self.assertFalse(slot.stream_enabled)
            self.assertEqual(slot.activity,0)
            for key in ('cpu_percent','ram_percent','gpu_percent','total_ram_percent','worker_ram_percent'):
                self.assertIsNone(getattr(slot,key))


if __name__ == '__main__':
    unittest.main()
