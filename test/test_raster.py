import numpy as np
import unittest

from raster_tools import Raster


def rs_eq_array(rs, ar):
    return (rs._rs.values == ar).all()


class TestRasterMath(unittest.TestCase):
    def setUp(self):
        self.rs1 = Raster("test/data/elevation.tif")
        self.rs1_np = self.rs1._rs.values
        self.rs2 = Raster("test/data/elevation2.tif")
        self.rs2_np = self.rs2._rs.values

    def tearDown(self):
        self.rs1.close()
        self.rs2.close()

    def test_add(self):
        # Raster + raster
        truth = self.rs1_np + self.rs2_np
        rst = self.rs1.add(self.rs2)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2.add(self.rs1)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs1 + self.rs2
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2 + self.rs1
        self.assertTrue(rs_eq_array(rst, truth))
        # Raster + scalar
        for v in [-23, 0, 1, 2, 321]:
            truth = self.rs1_np + v
            rst = self.rs1.add(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 + v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v + self.rs1
            self.assertTrue(rs_eq_array(rst, truth))
        for v in [-23.3, 0.0, 1.0, 2.0, 321.4]:
            truth = self.rs1_np + v
            rst = self.rs1.add(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 + v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v + self.rs1
            self.assertTrue(rs_eq_array(rst, truth))

    def test_subtract(self):
        # Raster - raster
        truth = self.rs1_np - self.rs2_np
        rst = self.rs1.subtract(self.rs2)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2.subtract(self.rs1)
        self.assertTrue(rs_eq_array(rst, -truth))
        rst = self.rs1 - self.rs2
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2 - self.rs1
        self.assertTrue(rs_eq_array(rst, -truth))
        # Raster - scalar
        for v in [-1359, 0, 1, 2, 42]:
            truth = self.rs1_np - v
            rst = self.rs1.subtract(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 - v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v - self.rs1
            self.assertTrue(rs_eq_array(rst, -truth))
        for v in [-1359.2, 0.0, 1.0, 2.0, 42.5]:
            truth = self.rs1_np - v
            rst = self.rs1.subtract(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 - v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v - self.rs1
            self.assertTrue(rs_eq_array(rst, -truth))

    def test_mult(self):
        # Raster * raster
        truth = self.rs1_np * self.rs2_np
        rst = self.rs1.multiply(self.rs2)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2.multiply(self.rs1)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs1 * self.rs2
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2 * self.rs1
        self.assertTrue(rs_eq_array(rst, truth))
        # Raster * scalar
        for v in [-123, 0, 1, 2, 345]:
            truth = self.rs1_np * v
            rst = self.rs1.multiply(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 * v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v * self.rs1
            self.assertTrue(rs_eq_array(rst, truth))
        for v in [-123.9, 0.0, 1.0, 2.0, 345.3]:
            truth = self.rs1_np * v
            rst = self.rs1.multiply(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 * v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v * self.rs1
            self.assertTrue(rs_eq_array(rst, truth))

    def test_div(self):
        # Raster / raster
        truth = self.rs1_np / self.rs2_np
        rst = self.rs1.divide(self.rs2)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2.divide(self.rs1)
        self.assertTrue(rs_eq_array(rst, 1 / truth))
        rst = self.rs1 / self.rs2
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2 / self.rs1
        self.assertTrue(rs_eq_array(rst, 1 / truth))
        # Raster / scalar, scalar / raster
        for v in [-123, -1, 1, 2, 345]:
            truth = self.rs1_np / v
            rst = self.rs1.divide(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 / v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v / self.rs1
            np.testing.assert_array_almost_equal(rst._rs.values, 1 / truth)
        for v in [-123.8, -1.0, 1.0, 2.0, 345.6]:
            truth = self.rs1_np / v
            rst = self.rs1.divide(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 / v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v / self.rs1
            np.testing.assert_array_almost_equal(rst._rs.values, 1 / truth)

    def test_power(self):
        # Raster ** raster
        rs1 = self.rs1 / self.rs1._rs.max() * 2
        rs2 = self.rs2 / self.rs2._rs.max() * 2
        rs1_np = self.rs1_np / self.rs1_np.max() * 2
        rs2_np = self.rs2_np / self.rs2_np.max() * 2
        truth = rs1_np ** rs2_np
        rst = rs1.pow(rs2)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = rs2.pow(rs1)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = rs1 ** rs2
        self.assertTrue(rs_eq_array(rst, truth))
        truth = rs2_np ** rs1_np
        rst = rs2 ** rs1
        self.assertTrue(rs_eq_array(rst, truth))
        # Raster ** scalar, scalar ** raster
        for v in [-10, -1, 1, 2, 11]:
            truth = rs1_np ** v
            rst = rs1.pow(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = rs1 ** v
            self.assertTrue(rs_eq_array(rst, truth))
            # Avoid complex numbers issues
            if v >= 0:
                truth = v ** rs1_np
                rst = v ** rs1
                self.assertTrue(rs_eq_array(rst, truth))
        for v in [-10.5, -1.0, 1.0, 2.0, 11.3]:
            truth = rs1_np ** v
            rst = rs1.pow(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = rs1 ** v
            self.assertTrue(rs_eq_array(rst, truth))
            # Avoid complex numbers issues
            if v >= 0:
                truth = v ** rs1_np
                rst = v ** rs1
                self.assertTrue(rs_eq_array(rst, truth))


if __name__ == "__main__":
    unittest.main()
