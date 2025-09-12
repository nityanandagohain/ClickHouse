-- Test that SHARED_ONLY paths don't count against max_dynamic_paths limit
-- SHARED_ONLY paths should bypass the dynamic path limit

DROP TABLE IF EXISTS test_json_shared_only_max_dynamic;

CREATE TABLE test_json_shared_only_max_dynamic (
    id UInt32,
    data JSON(
        max_dynamic_paths=2,
        user.id UInt32,
        SHARED_ONLY meta.timestamp
    )
) ENGINE = Memory;
-- Insert data with more paths than max_dynamic_paths limit
-- We have max_dynamic_paths=2, but we'll insert 5 dynamic paths
-- The SHARED_ONLY paths should not count against this limit
INSERT INTO test_json_shared_only_max_dynamic VALUES 
(1, '{"user": {"id": 1}, "meta": {"timestamp": "2023-01-01", "version": "1.0"}, "dynamic1": "value1", "dynamic2": "value2"}'),
(2, '{"user": {"id": 2}, "meta": {"timestamp": "2023-01-02", "version": "1.1"}, "dynamic1": "value6", "dynamic2": "value7"}');

-- Verify that all paths are accessible
SELECT 
    data.user.id as user_id,
    data.meta.timestamp as meta_timestamp,
    data.meta.version as meta_version,
    data.dynamic1 as dynamic1,
    data.dynamic2 as dynamic2
FROM test_json_shared_only_max_dynamic
ORDER BY id limit 1;


-- Verify shared and dynamic paths
SELECT 
    length(JSONDynamicPaths(data)) AS dynamic_path_count,
    length(JSONSharedDataPaths(data)) AS shared_path_count,
    JSONDynamicPaths(data) AS dynamic_paths,
    JSONSharedDataPaths(data) AS shared_paths
FROM test_json_shared_only_max_dynamic
ORDER BY id;



DROP TABLE test_json_shared_only_max_dynamic;
