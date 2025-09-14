-- Test basic SHARED_ONLY functionality
-- SHARED_ONLY paths should always be stored in shared data and never promoted to dynamic columns

DROP TABLE IF EXISTS test_json_shared_only_basic;

CREATE TABLE test_json_shared_only_basic (
    id UInt32,
    data JSON(
        max_dynamic_paths=10,
        a.b.c UInt32,
        SHARED_ONLY x.y,
        SHARED_ONLY meta.timestamp
    )
) ENGINE = Memory;

-- Insert data with SHARED_ONLY paths and regular paths
INSERT INTO test_json_shared_only_basic VALUES 
(1, '{"a": {"b": {"c": 100}}, "x": {"y": "shared_value"}, "meta": {"timestamp": "2023-01-01"}, "dynamic1": "value1"}'),
(2, '{"a": {"b": {"c": 200}}, "x": {"y": "another_shared"}, "meta": {"timestamp": "2023-01-02"}, "dynamic1": "value4"}');

-- Verify shared and dynamic paths
SELECT 
    length(JSONDynamicPaths(data)) AS dynamic_path_count,
    length(JSONSharedDataPaths(data)) AS shared_path_count,
    JSONDynamicPaths(data) AS dynamic_paths,
    JSONSharedDataPaths(data) AS shared_paths
FROM test_json_shared_only_basic
ORDER BY id;

-- Select the data to verify all paths work
SELECT
    data.a.b.c as a_b_c,
    data.x.y as x_y,
    data.meta.timestamp as meta_timestamp,
    data.dynamic1 as dynamic1
FROM test_json_shared_only_basic
ORDER BY id LIMIT 1;

DROP TABLE test_json_shared_only_basic;