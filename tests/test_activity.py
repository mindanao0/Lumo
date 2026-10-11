"""Behavioral checks for measured device activity and manual color overrides."""
import math
import unittest

from lumo.activity import device_activity
from lumo.showcase import ShowcaseState
from lumo.telemetry import ProcessReading,Snapshot


class DeviceActivityTests(unittest.TestCase):
    def test_measured_percentages_and_process_cpu_are_normalized(self):
        snapshot=Snapshot(cpu=75.,gpu=25.,cores=[0.]*8,ram_used=8.,ram_total=16.,
                          process_details=[ProcessReading(1,"worker",400.,0.)])
        values=device_activity(snapshot)
        self.assertEqual(values[0],.75)
        self.assertEqual(values[4],.25)
        self.assertEqual(values[3],.5)
        self.assertEqual(values[2],.5)

    def test_network_and_disk_light_increase_with_byte_rates(self):
        idle=device_activity(Snapshot())
        low=device_activity(Snapshot(net_down=2**20,disk_read=2**20))
        high=device_activity(Snapshot(net_down=2**26,disk_read=2**26))
        for index in (1,5):
            self.assertEqual(idle[index],0.)
            self.assertLess(low[index],high[index])
            self.assertLess(high[index],1.)

    def test_invalid_or_missing_sensor_readings_stay_finite(self):
        values=device_activity(Snapshot(cpu=math.nan,gpu=None,net_up=math.inf,
                                        disk_read=-10.,ram_used=0.,ram_total=0.))
        self.assertTrue(all(math.isfinite(value) and 0.<=value<=1. for value in values))

    def test_default_colors_are_reactive_and_manual_choice_overrides_them(self):
        state=ShowcaseState()
        self.assertTrue(state.reactive)
        state.update(72.)
        self.assertEqual(state.mode,"read")
        state.select("gpu")
        self.assertFalse(state.reactive)
        state.reactive=True
        self.assertTrue(state.reactive)


if __name__=="__main__":
    unittest.main()
