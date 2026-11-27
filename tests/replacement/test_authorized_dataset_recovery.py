import copy

import pytest
from test_dataset_update_wire import snapshot

from datalens_dev_mcp.api.errors import DataLensSafetyError, UncertainWriteError
from datalens_dev_mcp.objects.write import ObjectMutationService
from datalens_dev_mcp.operation_store import OperationStore


class DatasetBackend:
    def __init__(self):
        self.state = snapshot()
        self.calls = 0
        self.fail_at = {1}

    def object_get(self, kind, oid, *, branch="saved", revision_id=None):
        return {"ok": True, "identity": {"object_type": kind, "object_id": oid,
                "branch": branch, "revision_id": self.state["revId"]}, "object": copy.deepcopy(self.state)}

    def update(self, kind, oid, state):
        self.calls += 1
        if self.calls in self.fail_at:
            raise UncertainWriteError("synthetic lost acknowledgement")
        self.state = copy.deepcopy(state)
        self.state["revId"] = self.state["savedId"] = f"outer-r{self.calls}"
        self.state["dataset"]["revision_id"] = f"inner-r{self.calls}"
        return {"object_id": oid, "object": copy.deepcopy(self.state)}


def recovery(tmp_path):
    backend = DatasetBackend()
    writer = ObjectMutationService(reader=backend, backend=backend,
                                   store=OperationStore(tmp_path), provider_scope="synthetic-scope")
    change = {"object_type": "dataset", "object_id": "synthetic-dataset",
              "expected_revision": "outer-r1", "patch": {"dataset": {"description": "after"}}}
    unknown = writer.update_objects([change], operation_id="original")
    assert unknown["status"] == "uncertain"
    refs = [{"operation_id": "original", "object_id": "synthetic-dataset",
             "receipt_version": unknown["store_version"]}]
    return writer, backend, change, refs


def test_one_explicit_repeat_preserves_unknown_and_allows_fresh_cas_continuation(tmp_path):
    writer, backend, change, refs = recovery(tmp_path)
    original = writer.get_operation("original")
    with pytest.raises(DataLensSafetyError):
        writer.update_objects([change], operation_id="not-authorized")
    result = writer.update_objects([change], repeat_of=refs, operation_id="authorized")
    assert result["status"] == "completed"
    assert result["repeat_of"] == refs
    assert writer.get_operation("original")["results"][0]["effect_outcome"] == "unknown"
    assert writer.get_operation("original") == original
    assert writer.update_objects([change], repeat_of=refs, operation_id="authorized")["status"] == "completed"
    assert backend.calls == 2
    with pytest.raises(DataLensSafetyError):
        writer.update_objects([change], repeat_of=refs, operation_id="same-grant-again")
    following = copy.deepcopy(change)
    no_cas = copy.deepcopy(change)
    del no_cas["expected_revision"]
    with pytest.raises(DataLensSafetyError):
        writer.update_objects([no_cas], operation_id="unprotected-continuation")
    following["expected_revision"] = "outer-r2"
    following["patch"]["dataset"]["description"] = "next independent change"
    assert writer.update_objects([following], operation_id="following")["status"] == "completed"
    assert backend.calls == 3
    writer.store._compact_receipt(writer.store.get("authorized"))
    following["expected_revision"] = "outer-r3"
    following["patch"]["dataset"]["description"] = "after receipt compaction"
    assert writer.update_objects([following], operation_id="after-compaction")["status"] == "completed"


@pytest.mark.parametrize("defect", ["stale_receipt", "missing_cas", "wrong_target", "different_scope", "active_sender"])
def test_repeat_rejects_unproven_or_drifted_authority_without_dispatch(tmp_path, defect):
    writer, backend, change, refs = recovery(tmp_path)
    if defect == "stale_receipt":
        refs[0]["receipt_version"] -= 1
    elif defect == "missing_cas":
        del change["expected_revision"]
    elif defect == "wrong_target":
        refs[0]["object_id"] = "another-dataset"
    elif defect == "different_scope":
        writer.provider_scope = "another-scope"
    if defect == "active_sender":
        with writer.store.mutation_lease("original") as acquired:
            assert acquired
            with pytest.raises(DataLensSafetyError):
                writer.update_objects([change], repeat_of=refs, operation_id="repeat")
    else:
        with pytest.raises((DataLensSafetyError, ValueError)):
            writer.update_objects([change], repeat_of=refs, operation_id="repeat")
    assert backend.calls == 1


def test_stale_target_does_not_spend_a_grant_before_dispatch(tmp_path):
    writer, backend, change, refs = recovery(tmp_path)
    stale = copy.deepcopy(change)
    stale["expected_revision"] = "stale"
    result = writer.update_objects([stale], repeat_of=refs, operation_id="stale")
    assert result["results"][0]["code"] == "revision_changed"
    assert backend.calls == 1
    assert writer.update_objects([change], repeat_of=refs, operation_id="fresh")["status"] == "completed"


def test_another_unknown_requires_a_new_grant_covering_the_whole_chain(tmp_path):
    writer, backend, change, refs = recovery(tmp_path)
    backend.fail_at.add(2)
    second = writer.update_objects([change], repeat_of=refs, operation_id="second")
    assert second["status"] == "uncertain"
    with pytest.raises(DataLensSafetyError):
        writer.update_objects([change], repeat_of=refs, operation_id="reused-grant")
    refs.append({"operation_id": "second", "object_id": "synthetic-dataset",
                 "receipt_version": second["store_version"]})
    assert writer.update_objects([change], repeat_of=refs, operation_id="new-grant")["status"] == "completed"
    assert backend.calls == 3


@pytest.mark.parametrize("defect", ["ack", "create", "publish", "pending", "unscoped", "legacy_sender", "non_dataset"])
def test_repeat_does_not_release_other_holds(tmp_path, defect):
    writer, backend, change, refs = recovery(tmp_path)
    original = writer.store.get("original")
    if defect == "ack":
        original["results"][0]["write_returned"] = True
    elif defect in {"create", "publish"}:
        original["effect"] = defect
    elif defect == "pending":
        original["status"] = "pending"
    elif defect == "unscoped":
        original.pop("provider_scope")
    elif defect == "legacy_sender":
        original.pop("sender_contract")
    elif defect == "non_dataset":
        original["results"][0]["target"]["object_type"] = "wizard_chart"
    writer.store.put(original)
    refs[0]["receipt_version"] = original["store_version"]
    with pytest.raises(DataLensSafetyError):
        writer.update_objects([change], repeat_of=refs, operation_id="attempt")
    assert backend.calls == 1


def test_new_public_argument_is_validated_and_reaches_the_existing_writer(tmp_path, monkeypatch):
    from datalens_dev_mcp import server

    writer, backend, change, refs = recovery(tmp_path)
    monkeypatch.setattr(server, "default_mutation_service", lambda: writer)
    result = server.call_tool("dl_object_update", {"changes": [change], "repeat_of": refs,
                                                   "operation_id": "public-repeat"})
    assert result["structuredContent"]["status"] == "completed"
    assert backend.calls == 2
