import unittest
from touchline.metrics import Calibration,Point,movement_summary


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.cal=Calibration((1,0,0,0,1,0,0,0,1),'synthetic-reference',0,20,(0,0,100,100))
        self.points=[Point(t,2*t,0,'segment','synthetic-review') for t in (0,.5,1,1.5)]

    def test_known_constant_speed(self):
        result=movement_summary(self.points,self.cal)
        self.assertEqual(result['status'],'estimated')
        self.assertAlmostEqual(result['distance_m'],3)
        self.assertAlmostEqual(result['average_speed_mps'],2)
        self.assertAlmostEqual(result['maximum_windowed_speed_mps'],2)

    def test_missing_calibration_and_identity_are_unavailable(self):
        self.assertIsNone(movement_summary(self.points)['distance_m'])
        result=movement_summary([Point(0,0,0,'s'),Point(.5,1,0,'s')],self.cal)
        self.assertEqual(result['status'],'unavailable');self.assertIsNone(result['distance_m'])

    def test_no_gap_bridge_or_invented_peak(self):
        points=[Point(t,2*t,0,'s','review') for t in (0,.5,2,2.5)]
        result=movement_summary(points,self.cal)
        self.assertAlmostEqual(result['distance_m'],2)
        self.assertIsNone(result['maximum_windowed_speed_mps'])
        self.assertTrue(result['excluded_segments'])

    def test_segment_break_is_excluded(self):
        points=[Point(0,0,0,'a','review'),Point(.5,20,0,'b','review')]
        self.assertIsNone(movement_summary(points,self.cal)['distance_m'])

    def test_stationary_is_zero_when_actually_observed(self):
        points=[Point(t,1,1,'s','review') for t in (0,.5,1)]
        result=movement_summary(points,self.cal)
        self.assertEqual(result['distance_m'],0);self.assertEqual(result['maximum_windowed_speed_mps'],0)

    def test_duplicate_time_fails(self):
        with self.assertRaises(ValueError):movement_summary([self.points[0],self.points[0]],self.cal)

    def test_invalid_mapping_and_region(self):
        with self.assertRaises(ValueError):Calibration((0,)*9,'x',0,2,(0,0,1,1))
        result=movement_summary([Point(0,200,0,'s','r'),Point(.5,201,0,'s','r')],self.cal)
        self.assertIsNone(result['distance_m'])


if __name__=='__main__':unittest.main()
