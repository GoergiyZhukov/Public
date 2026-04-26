WITH user_monthly_stats AS (
    -- 1) CTE: агрегируем действия по месяцам
    SELECT 
        ua.user_id,
        DATE_TRUNC('month', ua.action_date) AS action_month,
        COUNT(*) AS actions_count,
        MIN(ua.action_date) OVER (PARTITION BY ua.user_id) AS first_action_date,
        MAX(ua.action_date) OVER (PARTITION BY ua.user_id) AS last_action_date
    FROM user_actions ua
    GROUP BY ua.user_id, DATE_TRUNC('month', ua.action_date)
),

ranked_users AS (
    -- 2) CTE: ранг пользователей внутри каждого месяца (кто сделал больше действий)
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY action_month ORDER BY actions_count DESC) AS rank_monthly,
        LAG(actions_count, 1) OVER (PARTITION BY user_id ORDER BY action_month) AS prev_month_actions,
        LEAD(actions_count, 1) OVER (PARTITION BY user_id ORDER BY action_month) AS next_month_actions
    FROM user_monthly_stats
)

-- 3) Финальный запрос: скользящее среднее за 3 месяца (оконная функция с ROWS)
SELECT 
    ru.user_id,
    ru.action_month,
    ru.actions_count,
    ru.first_action_date,
    ru.last_action_date,
    ru.rank_monthly,
    AVG(ru.actions_count) OVER (
        PARTITION BY ru.user_id 
        ORDER BY ru.action_month 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_3m,
    -- Когортный признак: месяц первого действия
    DATE_TRUNC('month', ru.first_action_date) AS cohort_month
FROM ranked_users ru
WHERE ru.rank_monthly <= 10   -- только топ-10 активных пользователей в месяц
ORDER BY ru.user_id, ru.action_month;
