# Copyright 2024 The Penzai Authors.\n
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# 
# """Tests for shapecheck."""
# 
import re
import textwrap

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np



class ShapecheckTest(absltest.TestCase):
    
      def test_same_empty(self):
            match = pz.chk.check_structure(value=[], pattern=[])
            self.assertEqual(dict(match), {})
            
            def test_same_objects(self):
                   match = pz.chk.check_structure(
                                 value=["a", (1, 2, 3), np.array([3, 5]), ("foo", "bar")],
                                                 pattern=["a", (1, 2, 3), np.array([3, 5]), pz.chk.ANY],
                                                         )
                   self.assertEqual(dict(match), {})
                   
                   def test_object_mismatches(self):
                        err = textwrap.dedent("""\\
                                                  Mismatch while checking structures:
                                                  At root[0]: Value \'a\' was not equal to the non-ArraySpec pattern \'b\'.
                                                  At root[1][0]: Value 1 was not equal to the non-ArraySpec pattern 3.
                                                  At root[1][2]: Value 3 was not equal to the non-ArraySpec pattern 1.
                                                  At root[2]: Value array([3, 5]) was not equal to the non-ArraySpec pattern array([3, 6]).
                                                  At root[3]: Value (\'foo\', \'bar\') was not equal to the non-ArraySpec pattern 100.
                                                  """).rstrip()
                        with self.assertRaisesWithLiteralMatch(pz.chk.StructureMismatchError, err):
                               pz.chk.check_structure(
                                               value=["a", (1, 2, 3), np.array([3, 5]), ("foo", "bar")],
                                                                   pattern=["b", (3, 2, 1), np.array([3, 6]), 100],
                                                                               )
                               
                               def test_simple_array(self):    match = pz.chk.check_structure(
                                                         value={"a": jax.ShapeDtypeStruct(shape=(1, 2, 3), dtype=jnp.float32)},
                                                                         pattern={"
