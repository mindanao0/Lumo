"""Behavioral checks for measured device activity and manual color overrides."""
import math
import unittest

from lumo.activity import device_activity, machine_cpu_percent, motion_activity, flow_speed
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

    def test_group_cpu_uses_whole_machine_capacity(self):
        self.assertEqual(machine_cpu_percent(200.,8),25.)
        self.assertEqual(machine_cpu_percent(float('nan'),8),0.)
        self.assertEqual(machine_cpu_percent(100.,0),100.)

    def test_ram_occupancy_does_not_accelerate_motion(self):
        idle=device_activity(Snapshot(cores=[0.]*8,ram_total=16.))
        full=device_activity(Snapshot(cores=[0.]*8,ram_total=16.,ram_used=16.))
        self.assertEqual(flow_speed(motion_activity(idle)),flow_speed(motion_activity(full)))
        busy=device_activity(Snapshot(cpu=90.,cores=[0.]*8))
        gpu=device_activity(Snapshot(gpu=90.,cores=[0.]*8))
        self.assertGreater(flow_speed(motion_activity(busy)),5*flow_speed(motion_activity(idle)))
        self.assertEqual(motion_activity(busy),motion_activity(gpu))

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
