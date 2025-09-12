-- Test ALTER TABLE with SHARED_ONLY paths
-- Should be able to add SHARED_ONLY patterns to existing tables

DROP TABLE IF EXISTS test_json_shared_only_alter;

CREATE TABLE test_json_shared_only_alter (
    id UInt32,
    data JSON(
        user.id UInt32,
        user.name String
    )
) ENGINE = Memory;

-- Insert initial data
INSERT INTO test_json_shared_only_alter VALUES 
(1, '{"user": {"id": 1, "name": "Alice"}, "meta": {"timestamp": "2023-01-01"}, "debug": {"level": "info"}}'),
(2, '{"user": {"id": 2, "name": "Bob"}, "meta": {"version": "1.0"}, "debug": {"trace": "abc123"}}');

-- Verify shared and dynamic paths
SELECT 
    length(JSONDynamicPaths(data)) AS dynamic_path_count,
    length(JSONSharedDataPaths(data)) AS shared_path_count,
    JSONDynamicPaths(data) AS dynamic_paths,
    JSONSharedDataPaths(data) AS shared_paths
FROM test_json_shared_only_alter
ORDER BY id;


-- Alter table to add SHARED_ONLY patterns
ALTER TABLE test_json_shared_only_alter MODIFY COLUMN data JSON(
    user.id UInt32,
    user.name String,
    SHARED_ONLY meta.timestamp,
    SHARED_ONLY REGEXP '^debug\\..*'
);

-- Insert more data after ALTER
INSERT INTO test_json_shared_only_alter VALUES 
(3, '{"user": {"id": 3, "name": "Charlie"}, "meta": {"timestamp": "2023-01-03"}, "debug": {"level": "debug", "trace": "xyz789"}}');

-- Verify shared and dynamic paths
SELECT 
    length(JSONDynamicPaths(data)) AS dynamic_path_count,
    length(JSONSharedDataPaths(data)) AS shared_path_count,
    JSONDynamicPaths(data) AS dynamic_paths,
    JSONSharedDataPaths(data) AS shared_paths
FROM test_json_shared_only_alter
ORDER BY id;

DROP TABLE test_json_shared_only_alter;
