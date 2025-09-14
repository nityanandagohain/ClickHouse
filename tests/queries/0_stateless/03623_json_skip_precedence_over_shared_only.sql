-- Test that SKIP takes precedence over SHARED_ONLY
-- When a path is both SKIP and SHARED_ONLY, it should be skipped entirely

DROP TABLE IF EXISTS test_json_skip_precedence;

CREATE TABLE test_json_skip_precedence (
    id UInt32,
    data JSON(
        max_dynamic_paths=2,
        user.id UInt32,
        SKIP sensitive.password,
        SHARED_ONLY sensitive.password,
        SHARED_ONLY sensitive.token,
        SKIP REGEXP '^temp\\..*',
        SHARED_ONLY REGEXP '^temp\\..*'
    )
) ENGINE = Memory;

-- Insert data with paths that are both SKIP and SHARED_ONLY
INSERT INTO test_json_skip_precedence VALUES 
(1, '{"user": {"id": 1}, "sensitive": {"password": "secret123", "token": "abc123"}, "temp": {"data": "temporary"}, "regular1": "value1"}'),
(2, '{"user": {"id": 2}, "sensitive": {"password": "secret456", "token": "def456"}, "temp": {"data": "temporary2"}, "regular1": "value2"}');

-- Verify that SKIP paths are not accessible (should return NULL)
SELECT 
    data.user.id as user_id,
    data.sensitive.password as password_should_be_null,
    data.sensitive.token as token_should_be_accessible,
    data.temp.data as temp_data_should_be_null,
    data.regular1 as regular1
FROM test_json_skip_precedence
ORDER BY id LIMIT 1;