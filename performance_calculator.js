function get_compount_nav_from_returns(returns, start_index) {
    let compound_nav = []
    for (let i = 0; i < returns.length; i++) {
        if (i <= start_index) {
            compound_nav.push(1.0)
        } else {
            compound_nav.push(compound_nav[i - 1] * (1 + returns[i]))
        }
    }
    return compound_nav
}

function get_sum_nav_from_returns(returns, start_index) {
    let sum_nav = []
    for (let i = 0; i < returns.length; i++) {
        if (i <= start_index) {
            sum_nav.push(1.0)
        } else {
            sum_nav.push(sum_nav[i - 1] + returns[i])
        }
    }
    return sum_nav
}

function get_max_draw_down_from_nav(nav) {
    let mdd = []
    let max = 1.0
    for (let i = 0; i < nav.length; i++) {
        if (nav[i] > max) {
            max = nav[i]
        }
        mdd.push(((nav[i] - max) / max) * 100)
    }
    return mdd
}

function get_win_rate_from_returns(returns) {
    let positive_count = 0
    for (let i = 0; i < returns.length; i++) {
        if (returns[i] > 0) {
            positive_count = positive_count + 1
        }
    }
    return positive_count / returns.length
}

function get_performance_statistics_from_returns(returns, start_index, end_index) {
    annual_working_days = 250
    selected_returns = returns.slice(start_index, end_index + 1)
    selected_compound_nav = get_compount_nav_from_returns(selected_returns, 0)
    total_return = selected_compound_nav[selected_compound_nav.length - 1] - 1
    annual_return = Math.pow(1 + total_return, (annual_working_days / selected_returns.length)) - 1
    annual_return_std = math.std(selected_returns) * Math.sqrt(annual_working_days)
    sharpe_ratio = annual_return / annual_return_std
    max_draw_down = math.min(get_max_draw_down_from_nav(selected_compound_nav)) * -1 / 100
    calmar_ratio = annual_return / max_draw_down
    win_rate = get_win_rate_from_returns(selected_returns)
    return [annual_return, annual_return_std, max_draw_down, sharpe_ratio, calmar_ratio, win_rate]
}

function create_performance_table(performances) {
    let table = document.createElement('table');
    let tableHead = document.createElement('thead');
    let tableBody = document.createElement('tbody');

    headers = ["", "annual_return", "vol", "max_dd", "sharpe_ratio", "calmar_ratio", "win_rate"]
    let header_row = document.createElement('tr');
    headers.forEach(function (header) {
        let cell = document.createElement('td');
        cell.appendChild(document.createTextNode(header));
        header_row.appendChild(cell);
    });
    tableHead.appendChild(header_row);

    performances.forEach(function (performance) {
        let row = document.createElement('tr');
        performance.forEach(function (cellData) {
            let cell = document.createElement('td');
            cell.appendChild(document.createTextNode(isNaN(cellData) ? cellData : cellData.toFixed(4)));
            row.appendChild(cell);
        });
        tableBody.appendChild(row);
    });

    table.appendChild(tableHead);
    table.appendChild(tableBody);

    return table
}