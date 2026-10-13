# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only SOURCE contract checks, not C++ registration or device proof."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'recovery/experiments/stageB_metadata_guard_v2'
CPP = (DIRECTORY / 'host.cpp').read_text()
BUILD = (DIRECTORY / 'build.py').read_text()
SPEC = importlib.util.spec_from_file_location('_stageB_cpp_guard_builder', DIRECTORY / 'build.py')
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def region(start, end):
    return CPP.split(start, 1)[1].split(end, 1)[0]


class SourceContract(unittest.TestCase):
    def test_no_device_or_queue_api(self):
        # These would change this candidate from a metadata-only host boundary.
        for forbidden in ('RunOpApi', 'recordStream(', 'synchronize(', 'aclrt',
                          'getCurrentNPUStream', 'getDefaultNPUStream', 'data_ptr<',
                          'copy_(', 'memcpy(', 'memset(', 'resize_(', 'set_(', 'empty_with_format('):
            # Strip C++ comments so ownership discussion is not treated as code.
            import re
            code = re.sub(r'//[^\n]*', '', CPP)
            self.assertNotIn(forbidden, code)

    def test_cpu_rejection_precedes_storage_cast(self):
        body = region('const torch_npu::NPUStorageDesc& descriptor', '\nvoid extent_check')
        self.assertLess(body.index('t.defined()'), body.index('static_cast<const torch_npu::NPUStorageImpl*>'))
        self.assertLess(body.index('DeviceType::PrivateUse1'), body.index('static_cast<const torch_npu::NPUStorageImpl*>'))
        self.assertIn('requires a defined NPU tensor', body)

    def test_snapshot_owns_tensor_and_original_storage(self):
        body = region('struct Snapshot {', '// Precheck before Snapshot')
        self.assertIn('Tensor tensor;', body)
        self.assertIn('c10::Storage storage;', body)
        self.assertIn('storage(value.storage())', body)
        self.assertIn('tensor_impl(value.unsafeGetTensorImpl())', body)
        self.assertIn('storage_impl(storage.unsafeGetStorageImpl())', body)

    def test_all_descriptor_and_light_fields_are_checked(self):
        body = region('void check(const Tensor& actual) const', '\n  void check()')
        for field in ('unsafeGetTensorImpl()', 'unsafeGetStorageImpl()', 'actual.sizes()',
                      'actual.strides()', 'actual.storage_offset()', 'actual.scalar_type()',
                      'actual.device()', 'actual.storage().nbytes()', 'actual.storage().data_ptr()',
                      'actual.const_data_ptr()', 'd.npu_format_', 'd.origin_format_', 'd.base_sizes_',
                      'd.base_strides_', 'd.storage_sizes_', 'version.check(actual)'):
            self.assertIn(field, body)

    def test_inference_version_not_faked_as_zero(self):
        body = region('struct Version {', '\nstruct Snapshot')
        self.assertIn('enabled ? counter.current_version() : 0', body)
        self.assertIn('counter.enabled() == enabled', body)
        self.assertIn('if (enabled) TORCH_CHECK(counter.current_version() == value', body)

    def test_per_layer_checks_all_nine_owners(self):
        body = region('bool validate(const Tensor& actual_parameter', '\n private:')
        for operation in ('producer_check(creator_)', 'parameter_.check(actual_parameter)',
                          'result_.check()', 'for (const auto& view : views_) view.check()', 'frame->validate()'):
            self.assertIn(operation, body)
        self.assertIn('const c10::intrusive_ptr<Frame>& frame', CPP)

    def test_frame_is_per_forward_and_entry_does_not_retain_frame(self):
        frame = region('class Frame :', '\nclass Entry :')
        entry = region('class Entry :', '} // namespace')
        self.assertIn('Snapshot projection_;', frame)
        self.assertIn('projection_.check()', frame)
        self.assertNotIn('Snapshot projection_', entry)
        self.assertNotIn('intrusive_ptr<Frame> frame_', entry)

    def test_view_alias_order_complete_owner_and_extent_overflow(self):
        body = region('Entry(Tensor parameter', '\n  bool validate(const Tensor& actual_parameter')
        for condition in ('views.size() == 6', 'view.storage().unsafeGetStorageImpl() == result_.storage_impl',
                          'view.storage_offset() == result_.offset + int64_t(i)*result_.stride[2]',
                          'view.strides() == at::IntArrayRef(result_.stride)'):
            self.assertIn(condition, body)
        extent = region('void extent_check', '\nstruct Version')
        for condition in ('physical <= capacity', 'INT64_MAX / t.stride(i)', 'high <= INT64_MAX-delta',
                          'uint64_t(high) < capacity'):
            self.assertIn(condition, extent)

    def test_typed_classes_register_only_required_api(self):
        binding = CPP.split('TORCH_LIBRARY(', 1)[1]
        self.assertTrue(binding.startswith('lingbot_stageB_guard_v2'))
        self.assertIn('m.class_<Frame>("Frame")', binding)
        self.assertIn('m.class_<Entry>("Entry")', binding)
        self.assertNotIn('def_pickle', CPP)
        self.assertNotIn('torch::init<int', CPP)

    def test_builder_import_is_stdlib_only(self):
        tree = ast.parse(BUILD)
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = [a.name.split('.')[0] for n in top for a in n.names] if top else []
        self.assertNotIn('torch', names)
        self.assertNotIn('torch_npu', names)
        self.assertEqual(BUILDER.SCHEMA, 'lingbot_stageB_guard_cpu_build_v2')

    def test_compile_terminal_is_durable_before_registration(self):
        self.assertLess(BUILD.index("durable_json(output / 'compile-result.json', terminal)"),
                        BUILD.index('torch.ops.load_library'))
        self.assertLess(BUILD.index("receipt['cpu_rejection_cases'].append"),
                        BUILD.index("receipt['registration'] = True"))
        self.assertIn("need(not torch_npu.npu.is_initialized()", BUILD)
        self.assertIn('exist_ok=False', BUILD)
        self.assertIn("if not args.execute:", BUILD)

    def test_atomic_json_order_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            events = []
            original_replace = BUILDER.os.replace
            def replace(a, z):
                events.append('replace')
                original_replace(a, z)
            with patch.object(BUILDER.os, 'fsync', side_effect=lambda fd: events.append('fsync')):
                with patch.object(BUILDER.os, 'replace', side_effect=replace):
                    BUILDER.durable_json(path, {'finished': True})
            self.assertEqual(events, ['fsync', 'replace', 'fsync'])
            self.assertFalse(path.with_name('receipt.json.tmp').exists())
            with self.assertRaises(ValueError):
                BUILDER.durable_json(Path(directory) / 'bad.json', {'x': float('nan')})

    def test_pin_resolves_path_and_actual_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source'
            path.write_bytes(b'abc')
            self.assertEqual(BUILDER.pin(path), {'path': str(path.resolve()), 'bytes': 3,
                'sha256': 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'})


if __name__ == '__main__': unittest.main()
