# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome import tensor_utils
from alphagenome.data import genome
from alphagenome.data import track_data
from alphagenome.protos import dna_model_pb2
import jaxtyping
import ml_dtypes
import numpy as np
import pandas as pd


class TrackDataInitTest(parameterized.TestCase):

  @parameterized.product(copy=[True, False], include_ontologies=[True, False])
  def test_valid_init_properties(self, copy: bool, include_ontologies: bool):
    values = np.arange(15).reshape((5, 3)).astype(np.int32)
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    if include_ontologies:
      metadata['ontology_curie'] = [
          'UBERON:0000005',
          'UBERON:0000007',
          None,
      ]
    tdata = track_data.TrackData(values, metadata)
    if copy:
      tdata = tdata.copy()
    self.assertListEqual(tdata.positional_axes, [0])
    self.assertEqual(tdata.num_tracks, 3)
    self.assertEqual(tdata.width, 5)
    self.assertEqual(tdata.resolution, 1)
    self.assertListEqual(list(tdata.names), ['first', 'first', 'second'])
    self.assertListEqual(list(tdata.strands), ['+', '-', '.'])
    if include_ontologies:
      self.assertEqual(
          [
              o.ontology_curie if o is not None else None
              for o in tdata.ontology_terms
          ],
          metadata['ontology_curie'].tolist(),
      )
    else:
      self.assertIsNone(tdata.ontology_terms)

  @parameterized.product(
      include_ontologies=[True, False],
      bytes_per_chunk=[4, 8],
      dtypes=[np.float32, np.int32],
      resolution=[1, 128],
      include_interval=[True, False],
      include_biosample=[True, False],
      include_transcription_factor=[True, False],
      include_histone_mark=[True, False],
      include_gtex_tissue=[True, False],
      include_endedness=[True, False],
      include_genetically_modified=[True, False],
      include_data_source=[True, False],
      override_interval=[True, False],
  )
  def test_proto_roundtrip(
      self,
      include_ontologies: bool,
      bytes_per_chunk: int,
      dtypes: np.dtype,
      resolution: int,
      include_interval: bool,
      include_biosample: bool,
      include_transcription_factor: bool,
      include_histone_mark: bool,
      include_gtex_tissue: bool,
      include_endedness: bool,
      include_genetically_modified: bool,
      include_data_source: bool,
      override_interval: bool,
  ):
    df = pd.DataFrame({
        'name': [f'track_{i}' for i in range(7)],
        'strand': ['.', '+', '-', '.', '+', '-', '.'],
        'Assay title': ['foo', 'bar', 'baz', 'qux', 'quux', 'wuz', 'buz'],
    })
    if include_ontologies:
      df['ontology_curie'] = [
          'UBERON:0000005',
          'UBERON:0000007',
          'UBERON:0000007',
          'UBERON:0000007',
          None,
          None,
          None,
      ]
    if include_biosample:
      df['biosample_name'] = [f'biosample_{i}' for i in range(6)] + [None]
      df['biosample_type'] = [
          'primary_cell',
          'in_vitro_differentiated_cells',
          'cell_line',
          'tissue',
          'technical_sample',
          'organoid',
          None,
      ]
      df['biosample_life_stage'] = [
          'adult',
          'adult',
          None,
          'newborn',
          'child',
          'embryonic',
          None,
      ]
    if include_transcription_factor:
      df['transcription_factor'] = [
          'BCLAF1',
          'BRF2',
          'CBFA2T3',
          'CEBPB',
          'COPS2',
          'POLR2AphosphoS5',
          None,
      ]
    if include_histone_mark:
      df['histone_mark'] = [
          'H2AFZ',
          'H2AK5AC',
          'H2BK120AC',
          'H2BK12AC',
          'H2BK20AC',
          'H2BK5AC',
          None,
      ]
    if include_gtex_tissue:
      df['gtex_tissue'] = [
          'Liver',
          '',
          '',
          'Brain_Hippocampus',
          'Pancreas',
          'Artery_Aorta',
          None,
      ]
    if include_data_source:
      df['data_source'] = [
          'encode',
          'encode',
          'encode',
          'fantom',
          'fantom',
          'gtex',
          None,
      ]
    if include_endedness:
      df['endedness'] = [
          'paired',
          'single',
          'paired',
          'single',
          'paired',
          'single',
          None,
      ]
    if include_genetically_modified:
      df['genetically_modified'] = [
          True,
          False,
          True,
          False,
          True,
          False,
          None,
      ]

    values = np.array(
        [
            [0.0, 0.2, 0.3, 0.4, 0.5, 0.1, 0.1],
            [0.1, 0.3, 0.5, 0.6, 0.7, 0.1, 0.1],
            [0.2, 0.4, 0.7, 0.8, 0.9, 0.1, 0.1],
        ],
        dtype=dtypes,
    )
    expected_interval = (
        genome.Interval('chr1', 0, values.shape[0] * resolution)
        if include_interval or override_interval
        else None
    )

    metadata = track_data.TrackData(
        values=values,
        metadata=df,
        resolution=resolution,
        interval=genome.Interval('chr1', 0, values.shape[0] * resolution)
        if include_interval
        else None,
    )

    proto, chunks = metadata.to_protos(bytes_per_chunk=bytes_per_chunk)
    roundtrip = track_data.from_protos(
        proto,
        chunks,
        interval=genome.Interval('chr1', 0, values.shape[0] * resolution)
        if override_interval
        else None,
    )

    np.testing.assert_array_equal(roundtrip.values, values)
    pd.testing.assert_frame_equal(roundtrip.metadata, df)
    self.assertEqual(roundtrip.resolution, resolution)
    self.assertEqual(roundtrip.interval, expected_interval)

  @parameterized.product(
      bytes_per_chunk=[0, 4, 8], dtype=[ml_dtypes.bfloat16, np.float16]
  )
  def test_from_low_precision_proto(self, bytes_per_chunk, dtype):
    values = np.array(
        [
            [0.0, 0.2, 0.3, 0.4, 0.5],
            [0.1, 0.3, 0.5, 0.6, 0.7],
            [0.2, 0.4, 0.7, 0.8, 0.9],
        ],
        dtype=dtype,
    )
    metadata = pd.DataFrame({
        'name': ['track_0', 'track_1', 'track_2', 'track_3', 'track_4'],
        'strand': ['.'] * 5,
    })
    tensor, chunks = tensor_utils.pack_tensor(
        values, bytes_per_chunk=bytes_per_chunk
    )
    proto = dna_model_pb2.TrackData(
        values=tensor,
        metadata=track_data.metadata_to_proto(metadata).metadata,
    )
    roundtrip = track_data.from_protos(proto, chunks)
    np.testing.assert_array_equal(roundtrip.values, values.astype(np.float32))
    pd.testing.assert_frame_equal(roundtrip.metadata, metadata)

  @parameterized.parameters([
      dict(num_positional=0),
      dict(num_positional=1),
      dict(num_positional=2),
  ])
  def test_valid_positional_axis(self, num_positional: int):
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.int32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(values, metadata)
    self.assertLen(tdata.positional_axes, num_positional)

  def test_invalid_init(self):
    values = np.arange(12).reshape((2, 2, 3)).astype(np.int32)
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    with self.assertRaises(ValueError):
      # Missing strand
      track_data.TrackData(values, metadata[['name', 'other']])
    with self.assertRaises(ValueError):
      # Missing name
      track_data.TrackData(values, metadata[['strand', 'other']])
    with self.assertRaises(ValueError):
      # Not matching number of tracks
      track_data.TrackData(values[:, :, :2], metadata)
    with self.assertRaises(ValueError):
      # Not matching positional axis
      track_data.TrackData(values[:1], metadata)
    with self.assertRaises(ValueError):
      # Not matching interval width
      track_data.TrackData(
          values, metadata, interval=genome.Interval('chr1', 0, 3)
      )

  def test_invalid_value_types_raises(self):
    values = np.array(['foo', 'bar', 'baz'], dtype=str)
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    with self.assertRaises(jaxtyping.TypeCheckError):
      track_data.TrackData(values, metadata)


class TrackDataSlicingTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(num_positional=0),
      dict(num_positional=1),
      dict(num_positional=2),
  ])
  def test_slice_by_positions(self, num_positional: int):
    # Values are [0,1,2] if num_positional=0, [[0,1,2],[3,4,5]] if
    # num_positional=1, etc.
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.int32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    interval = genome.Interval('chr1', 0, 4)
    tdata = track_data.TrackData(
        values, metadata, resolution=2, interval=interval
    )
    assert tdata.interval is not None
    self.assertIsNotNone(tdata.interval)
    self.assertEqual(tdata.interval.start, 0)
    self.assertEqual(tdata.interval.end, 4)
    # Note that track data with no positional axes has a width of 0 and so
    # cannot be sliced in this way.
    if num_positional:
      # Slicing to the full width (no-op).
      tdata2 = tdata.slice_by_positions(0, 4)
      assert tdata2.interval is not None
      self.assertEqual(tdata2.width, tdata.width)
      # Slicing to half the width.
      tdata2 = tdata.slice_by_positions(0, 2)
      self.assertEqual(tdata2.values.shape, tuple([1] * num_positional + [3]))
      self.assertEqual(tdata2.interval.start, 0)
      self.assertEqual(tdata2.interval.end, 2)
      self.assertEqual(tdata2.width, tdata2.interval.width)
      self.assertEqual(tdata2.width, 2)
      # Fails because (end - start) is not evenly divisible by resolution.
      with self.assertRaises(ValueError):
        tdata.slice_by_positions(0, 3)
      with self.assertRaises(ValueError):
        tdata.slice_by_positions(0, 1)
    # Fails because there are no positional axes to slice into.
    else:
      self.assertEqual(tdata.width, 0)
      with self.assertRaises(ValueError):
        tdata.slice_by_positions(0, 4)

  @parameterized.parameters([
      dict(num_positional=0),
      dict(num_positional=1),
      dict(num_positional=2),
  ])
  def test_slice_by_interval(self, num_positional: int):
    # Values are [0,1,2] if num_positional=0, [[0,1,2],[3,4,5]] if
    # num_positional=1, etc.
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.int32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    interval = genome.Interval('chr1', 1, 5)
    tdata = track_data.TrackData(
        values, metadata, resolution=2, interval=interval
    )
    assert tdata.interval is not None

    if num_positional:
      # Slicing to the full width (no-op).
      tdata2 = tdata.slice_by_interval(interval)
      assert tdata2.interval is not None
      self.assertEqual(tdata2.width, tdata.width)
      # Slicing to half the width.
      tdata2 = tdata.slice_by_interval(genome.Interval('chr1', 1, 3))
      self.assertEqual(tdata2.values.shape, tuple([1] * num_positional + [3]))
      self.assertEqual(tdata2.interval.start, 1)
      self.assertEqual(tdata2.interval.end, 3)
      self.assertEqual(tdata2.width, tdata2.interval.width)
      self.assertEqual(tdata2.width, 2)
      # Fails because (end - start) is not evenly divisible by resolution.
      with self.assertRaises(ValueError):
        tdata.slice_by_interval(genome.Interval('chr1', 1, 4))
      with self.assertRaises(ValueError):
        tdata.slice_by_interval(genome.Interval('chr1', 1, 2))

      # Same would work if we set match_resolution=True.
      tdata2 = tdata.slice_by_interval(
          genome.Interval('chr1', 1, 4), match_resolution=True
      )
      self.assertEqual(tdata2.interval.start, 1)
      self.assertEqual(tdata2.interval.end, 5)
      tdata2 = tdata.slice_by_interval(
          genome.Interval('chr1', 2, 5), match_resolution=True
      )
      self.assertEqual(tdata2.interval.start, 1)
      self.assertEqual(tdata2.interval.end, 5)
      tdata2 = tdata.slice_by_interval(
          genome.Interval('chr1', 3, 5), match_resolution=True
      )
      self.assertEqual(tdata2.interval.start, 3)
      self.assertEqual(tdata2.interval.end, 5)
    else:
      # Fails because there are no positional axes to slice into.
      self.assertEqual(tdata.width, 0)
      with self.assertRaises(ValueError):
        tdata.slice_by_interval(interval)

  @parameterized.parameters([
      dict(num_positional=0),
      dict(num_positional=1),
      dict(num_positional=2),
  ])
  def test_pad(self, num_positional: int):
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.int32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(values, metadata, resolution=2)
    if num_positional:
      self.assertEqual(tdata.width, 4)
    tdata2 = tdata.pad(0, 0)
    self.assertEqual(tdata2.width, tdata.width)
    tdata2 = tdata.pad(2, 0)
    self.assertEqual(tdata2.values.shape, tuple([3] * num_positional + [3]))
    if num_positional:
      self.assertEqual(tdata2.width, 6)
    with self.assertRaises(ValueError):
      tdata.pad(1, 1)

  @parameterized.parameters([
      dict(num_positional=0),
      dict(num_positional=1),
      dict(num_positional=2),
  ])
  def test_resize(self, num_positional: int):
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.int32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(
        values,
        metadata,
        resolution=2,
        interval=(genome.Interval('chr1', 0, 4) if num_positional else None),
    )
    if num_positional:
      self.assertEqual(tdata.width, 4)
    tdata2 = tdata.resize(4)
    self.assertEqual(tdata2.width, tdata.width)
    tdata2 = tdata.resize(6)
    self.assertEqual(tdata2.values.shape, tuple([3] * num_positional + [3]))
    if num_positional:
      self.assertEqual(tdata2.width, 6)
    tdata2 = tdata.resize(2)
    self.assertEqual(tdata2.values.shape, tuple([1] * num_positional + [3]))
    if num_positional:
      self.assertEqual(tdata2.width, 2)
    # Pad and crop to reach the same original array.
    tdata2 = tdata.resize(6).resize(4)
    if num_positional:
      self.assertEqual(tdata2.width, 4)
    self.assertTrue(np.all(tdata2.values == tdata.values))
    if num_positional:
      with self.assertRaises(ValueError):
        tdata.resize(5)


class TrackDataUpDownSampleTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(num_positional=0, aggregation_type=track_data.AggregationType.SUM),
      dict(num_positional=1, aggregation_type=track_data.AggregationType.SUM),
      dict(num_positional=2, aggregation_type=track_data.AggregationType.SUM),
      dict(num_positional=0, aggregation_type=track_data.AggregationType.MAX),
      dict(num_positional=1, aggregation_type=track_data.AggregationType.MAX),
      dict(num_positional=2, aggregation_type=track_data.AggregationType.MAX),
  ])
  def test_upsample(
      self, num_positional: int, aggregation_type: track_data.AggregationType
  ):
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.float32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(values, metadata, resolution=2)
    tdata2 = tdata.upsample(resolution=1, aggregation_type=aggregation_type)
    self.assertEqual(tdata2.width, tdata.width)
    self.assertEqual(tdata2.values.shape, tuple([4] * num_positional + [3]))
    if aggregation_type == track_data.AggregationType.SUM:
      self.assertEqual(tdata2.values.sum(), tdata.values.sum())
    elif aggregation_type == track_data.AggregationType.MAX:
      self.assertEqual(tdata2.values.max(), tdata.values.max())
    with self.assertRaises(ValueError):
      tdata.upsample(resolution=4)

  @parameterized.parameters([
      dict(
          num_positional=0,
          aggregation_type=track_data.AggregationType.SUM,
      ),
      dict(
          num_positional=1,
          aggregation_type=track_data.AggregationType.SUM,
      ),
      dict(
          num_positional=2,
          aggregation_type=track_data.AggregationType.SUM,
      ),
      dict(
          num_positional=0,
          aggregation_type=track_data.AggregationType.MAX,
      ),
      dict(
          num_positional=1,
          aggregation_type=track_data.AggregationType.MAX,
      ),
      dict(
          num_positional=2,
          aggregation_type=track_data.AggregationType.MAX,
      ),
  ])
  def test_downsample(
      self, num_positional: int, aggregation_type: track_data.AggregationType
  ):
    values = (
        np.arange(4**num_positional * 3)
        .reshape([4] * num_positional + [3])
        .astype(np.float32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(values, metadata, resolution=1)
    tdata2 = tdata.downsample(resolution=2, aggregation_type=aggregation_type)
    self.assertEqual(tdata2.width, tdata.width)
    self.assertEqual(tdata2.values.shape, tuple([2] * num_positional + [3]))
    if aggregation_type == track_data.AggregationType.SUM:
      self.assertEqual(tdata2.values.sum(), tdata.values.sum())
    elif aggregation_type == track_data.AggregationType.MAX:
      self.assertEqual(tdata2.values.max(), tdata.values.max())
    tdata = track_data.TrackData(values, metadata, resolution=2)
    with self.assertRaises(ValueError):
      tdata.downsample(resolution=1)

  @parameterized.parameters([
      dict(
          num_positional=0,
          resolution=1,
          aggregation_type=track_data.AggregationType.SUM,
      ),
      dict(
          num_positional=1,
          resolution=1,
          aggregation_type=track_data.AggregationType.SUM,
      ),
      dict(
          num_positional=2,
          resolution=1,
          aggregation_type=track_data.AggregationType.SUM,
      ),
      dict(
          num_positional=0,
          resolution=4,
          aggregation_type=track_data.AggregationType.SUM,
      ),
      dict(
          num_positional=1,
          resolution=4,
          aggregation_type=track_data.AggregationType.SUM,
      ),
      dict(
          num_positional=2,
          resolution=4,
          aggregation_type=track_data.AggregationType.SUM,
      ),
      dict(
          num_positional=0,
          resolution=1,
          aggregation_type=track_data.AggregationType.MAX,
      ),
      dict(
          num_positional=1,
          resolution=1,
          aggregation_type=track_data.AggregationType.MAX,
      ),
      dict(
          num_positional=2,
          resolution=1,
          aggregation_type=track_data.AggregationType.MAX,
      ),
      dict(
          num_positional=0,
          resolution=4,
          aggregation_type=track_data.AggregationType.MAX,
      ),
      dict(
          num_positional=1,
          resolution=4,
          aggregation_type=track_data.AggregationType.MAX,
      ),
      dict(
          num_positional=2,
          resolution=4,
          aggregation_type=track_data.AggregationType.MAX,
      ),
  ])
  def test_change_resolution(
      self,
      num_positional: int,
      resolution: int,
      aggregation_type: track_data.AggregationType,
  ):
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.float32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(values, metadata, resolution=2)
    tdata2 = tdata.change_resolution(
        resolution=resolution, aggregation_type=aggregation_type
    )
    self.assertEqual(tdata2.width, tdata.width)
    self.assertEqual(
        tdata2.values.shape,
        tuple([tdata2.width // resolution] * num_positional + [3]),
    )
    if aggregation_type == track_data.AggregationType.SUM:
      self.assertEqual(tdata2.values.sum(), tdata.values.sum())
    elif aggregation_type == track_data.AggregationType.MAX:
      self.assertEqual(tdata2.values.max(), tdata.values.max())


class TrackDataTrackSubsetTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(num_positional=0),
      dict(num_positional=1),
      dict(num_positional=2),
  ])
  def test_filter_tracks(self, num_positional: int):
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.int32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(values, metadata, resolution=2)
    tdata2 = tdata.filter_tracks([False, True, False])
    self.assertEqual(tdata2.num_tracks, 1)
    self.assertListEqual(list(tdata2.names), ['first'])
    self.assertListEqual(list(tdata2.strands), ['-'])
    self.assertEqual(tdata2.metadata['other'].iloc[0], 'foo')
    self.assertEqual(tdata2.values.shape, tuple([2] * num_positional + [1]))

  @parameterized.parameters([
      dict(num_positional=0, strand='+'),
      dict(num_positional=1, strand='+'),
      dict(num_positional=2, strand='+'),
      dict(num_positional=0, strand='-'),
      dict(num_positional=1, strand='-'),
      dict(num_positional=2, strand='-'),
      dict(num_positional=0, strand='.'),
      dict(num_positional=1, strand='.'),
      dict(num_positional=2, strand='.'),
  ])
  def test_filter_to_strand(self, num_positional: int, strand: str):
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.int32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(values, metadata, resolution=2)
    if strand == '+':
      tdata2 = tdata.filter_to_positive_strand()
      name = 'first'
      other = 'foo'
    elif strand == '-':
      tdata2 = tdata.filter_to_negative_strand()
      name = 'first'
      other = 'foo'
    elif strand == '.':
      tdata2 = tdata.filter_to_unstranded()
      name = 'second'
      other = 'bar'
    else:
      raise ValueError(f'Invalid strand: {strand}')
    self.assertEqual(tdata2.num_tracks, 1)
    self.assertListEqual(list(tdata2.names), [name])
    self.assertListEqual(list(tdata2.strands), [strand])
    self.assertEqual(tdata2.metadata['other'].iloc[0], other)
    self.assertEqual(tdata2.values.shape, tuple([2] * num_positional + [1]))

  @parameterized.parameters([
      dict(num_positional=0),
      dict(num_positional=1),
      dict(num_positional=2),
  ])
  def test_filter_nonpositive_nonnegative_strand(self, num_positional: int):
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.int32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(values, metadata, resolution=2)
    tdata2 = tdata.filter_to_nonpositive_strand()
    self.assertListEqual(list(tdata2.strands), ['-', '.'])
    tdata2 = tdata.filter_to_nonnegative_strand()
    self.assertListEqual(list(tdata2.strands), ['+', '.'])

  @parameterized.parameters([
      dict(num_positional=0),
      dict(num_positional=1),
      dict(num_positional=2),
  ])
  def test_filter_to_stranded(self, num_positional: int):
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.int32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(values, metadata, resolution=2)
    tdata2 = tdata.filter_to_stranded()
    self.assertEqual(tdata2.num_tracks, 2)
    self.assertListEqual(list(tdata2.names), ['first', 'first'])
    self.assertListEqual(list(tdata2.strands), ['+', '-'])
    self.assertEqual(tdata2.metadata['other'].iloc[0], 'foo')
    self.assertEqual(tdata2.values.shape, tuple([2] * num_positional + [2]))

  @parameterized.parameters([
      dict(num_positional=0),
      dict(num_positional=1),
      dict(num_positional=2),
  ])
  def test_select_tracks_by_index(self, num_positional: int):
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.int32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(values, metadata, resolution=2)
    tdata2 = tdata.select_tracks_by_index([1])
    self.assertEqual(tdata2.num_tracks, 1)
    self.assertListEqual(list(tdata2.names), ['first'])
    self.assertListEqual(list(tdata2.strands), ['-'])
    self.assertEqual(tdata2.metadata['other'].iloc[0], 'foo')
    self.assertEqual(tdata2.values.shape, tuple([2] * num_positional + [1]))

  @parameterized.parameters([
      dict(num_positional=0),
      dict(num_positional=1),
      dict(num_positional=2),
  ])
  def test_select_tracks_by_name(self, num_positional: int):
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.int32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(values, metadata, resolution=2)
    tdata2 = tdata.select_tracks_by_name(['first'])
    self.assertEqual(tdata2.num_tracks, 2)
    self.assertListEqual(list(tdata2.names), ['first', 'first'])
    self.assertListEqual(list(tdata2.strands), ['+', '-'])
    self.assertEqual(tdata2.metadata['other'].iloc[0], 'foo')
    self.assertEqual(tdata2.values.shape, tuple([2] * num_positional + [2]))
    tdata2 = tdata.select_tracks_by_name(['second'])
    self.assertEqual(tdata2.num_tracks, 1)
    self.assertListEqual(list(tdata2.names), ['second'])
    self.assertListEqual(list(tdata2.strands), ['.'])
    self.assertEqual(tdata2.metadata['other'].iloc[0], 'bar')
    self.assertEqual(tdata2.values.shape, tuple([2] * num_positional + [1]))

  @parameterized.parameters([
      dict(num_positional=0),
      dict(num_positional=1),
      dict(num_positional=2),
  ])
  def test_groupby(self, num_positional: int):
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.int32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(values, metadata, resolution=2)
    tdata2_dict = tdata.groupby('other')
    self.assertListEqual(list(tdata2_dict.keys()), ['foo', 'bar'])
    self.assertListEqual(list(tdata2_dict['foo'].names), ['first', 'first'])
    self.assertListEqual(list(tdata2_dict['bar'].names), ['second'])
    self.assertEqual(tdata2_dict['foo'].num_tracks, 2)
    self.assertEqual(tdata2_dict['bar'].num_tracks, 1)
    self.assertEqual(tdata2_dict['foo'].width, tdata.width)
    self.assertEqual(tdata2_dict['bar'].width, tdata.width)


class TrackDataReverseComplementTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(num_positional=0),
      dict(num_positional=1),
      dict(num_positional=2),
  ])
  def test_reverse_complement(self, num_positional: int):
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.int32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(values, metadata, resolution=2)
    tdata2 = tdata.reverse_complement()
    if num_positional == 0:
      self.assertTrue(np.all(tdata2.values[[1, 0, 2]] == tdata.values))
    elif num_positional == 1:
      self.assertTrue(np.all(tdata2.values[::-1, [1, 0, 2]] == tdata.values))
    elif num_positional == 2:
      self.assertTrue(
          np.all(tdata2.values[::-1, ::-1, [1, 0, 2]] == tdata.values)
      )
    self.assertListEqual(list(tdata2.names), list(tdata.names))
    self.assertListEqual(list(tdata2.strands), ['-', '+', '.'])

  @parameterized.parameters([
      dict(num_positional=0),
      dict(num_positional=1),
      dict(num_positional=2),
  ])
  def test_reverse_complement_raises(self, num_positional: int):
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.int32)
    )
    metadata = pd.DataFrame({
        'name': ['first', 'third', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(values, metadata, resolution=2)
    with self.assertRaises(ValueError):
      # Not matching number of strands by name
      tdata.reverse_complement()


class TrackDataConcatTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(num_positional=0),
      dict(num_positional=1),
      dict(num_positional=2),
      dict(num_positional=0, new_metadata_values=['tdata1', 'tdata2']),
      dict(num_positional=0, new_metadata_values=['tdata1', None]),
  ])
  def test_concat(
      self, num_positional: int, new_metadata_values: list[str] | None = None
  ):
    values = (
        np.arange(2**num_positional * 3)
        .reshape([2] * num_positional + [3])
        .astype(np.int32)
    )
    metadata1 = pd.DataFrame({
        'name': ['first', 'first', 'second'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata = track_data.TrackData(values, metadata1, resolution=2, uns={'a': 1})
    metadata2 = pd.DataFrame({
        'name': ['third', 'third', 'fourth'],
        'strand': ['+', '-', '.'],
        'other': ['foo', 'foo', 'bar'],
    })
    tdata2 = track_data.TrackData(values, metadata2, resolution=2, uns={'a': 1})
    with self.assertRaises(ValueError):
      track_data.concat([tdata, tdata])
    tdata3 = track_data.concat(
        [tdata, tdata2],
        ('new_column', new_metadata_values) if new_metadata_values else None,
    )
    self.assertEqual(tdata3.num_tracks, 6)
    self.assertEqual(tdata3.width, tdata.width)
    self.assertIsNone(tdata3.uns)
    self.assertEqual(tdata3.values.shape, tuple([2] * num_positional + [6]))
    self.assertLen(tdata3.metadata, 6)
    if new_metadata_values:
      # Expect error if the column already exists.
      with self.assertRaises(ValueError):
        track_data.concat([tdata, tdata2], ('name', new_metadata_values))
      self.assertSequenceEqual(
          list(tdata3.metadata['new_column'].unique()), new_metadata_values
      )
    else:
      self.assertNotIn('new_column', tdata3.metadata.columns)


class TrackDataInterleaveTest(absltest.TestCase):

  def test_interleave(self):
    values_1 = np.array(
        [
            [0.0, 0.2, 0.3, 0.4, 0.5],
            [0.1, 0.3, 0.5, 0.6, 0.7],
            [0.2, 0.4, 0.7, 0.8, 0.9],
        ],
        dtype=np.float32,
    )
    metadata_1 = pd.DataFrame({
        'name': ['track_0', 'track_1', 'track_2', 'track_3', 'track_4'],
        'strand': '.',
        'gtex_tissue': ['BLADDER', 'LIVER', 'LIVER', 'STOMACH', 'STOMACH'],
        'padding': [False, False, False, False, True],
    })
    values_2 = -1 * values_1
    metadata_2 = metadata_1.copy()
    data_1 = track_data.TrackData(values=values_1, metadata=metadata_1)
    data_2 = track_data.TrackData(values=values_2, metadata=metadata_2)
    track_datas = [data_1, data_2]
    name_prefixes = ['REF_', 'ALT_']
    interleaved_track_data = track_data.interleave(track_datas, name_prefixes)
    expected_values = np.array(
        [
            [0.0, -0.0, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5],
            [0.1, -0.1, 0.3, -0.3, 0.5, -0.5, 0.6, -0.6, 0.7, -0.7],
            [0.2, -0.2, 0.4, -0.4, 0.7, -0.7, 0.8, -0.8, 0.9, -0.9],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        interleaved_track_data.values, expected_values
    )
    expected_metadata = pd.DataFrame({
        'name': [
            'REF_track_0',
            'ALT_track_0',
            'REF_track_1',
            'ALT_track_1',
            'REF_track_2',
            'ALT_track_2',
            'REF_track_3',
            'ALT_track_3',
            'REF_track_4',
            'ALT_track_4',
        ],
        'strand': ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
        'gtex_tissue': [
            'BLADDER',
            'BLADDER',
            'LIVER',
            'LIVER',
            'LIVER',
            'LIVER',
            'STOMACH',
            'STOMACH',
            'STOMACH',
            'STOMACH',
        ],
        'padding': [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
        ],
    })
    pd.testing.assert_frame_equal(
        interleaved_track_data.metadata, expected_metadata
    )


if __name__ == '__main__':
  absltest.main()
