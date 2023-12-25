var {{chart_id}} = echarts.init(document.getElementById("{{chart_id}}"), null, { renderer: 'svg' });
var {{chart_id}}_dynamic_content = document.getElementById("{{chart_id}}_dynamic_content");

{{chart_id}}_type = "{{ nav_type }}"

{{chart_id}}_legend_labels = []
{% for data in datasets %}
{{chart_id}}_legend_labels.push({{data["label"]}});
{% endfor %}

{{chart_id}}_x_labels = {{labels}};

{{chart_id}}_returns = {};
{% for data in datasets %}
{{chart_id}}_returns[{{data["label"]}}] = {{data["data"]}};
{% endfor %}

function {{chart_id}}_get_series(start_index) {
    let series = [];
    {{chart_id}}_legend_labels.forEach(label => {
        let nav;
        if({{chart_id}}_type == "compound") {
            nav = get_compount_nav_from_returns({{chart_id}}_returns[label], start_index);
        } else if ({{chart_id}}_type == "sum") {
            nav = get_sum_nav_from_returns({{chart_id}}_returns[label], start_index);
        } else {
            throw 'Invalid NAV type ' + {{chart_id}}_type;
        }
        series.push({
            type: 'line',
            name: label,
            data: nav
        });
    });
    return series;
};

function {{chart_id}}_clear_performance_statistics() {
    while ({{chart_id}}_dynamic_content.firstChild) {
        {{chart_id}}_dynamic_content.firstChild.remove();
    }
};

function {{chart_id}}_update_performance_statistics(start_index, end_index) {
    let performances = [];
    {{chart_id}}_legend_labels.forEach(label => {
        performance_statistics = get_performance_statistics_from_returns({{chart_id}}_returns[label], start_index, end_index);
        performances.push([label, ...performance_statistics]);
    });
    {{chart_id}}_dynamic_content.appendChild(create_performance_table(performances));
};

{{chart_id}}.setOption({
    tooltip: {
      trigger: 'axis'
    },
    legend: {
        data: {{chart_id}}_legend_labels
    },
    grid: {
        top: '20%',
        bottom: 35
    },
    toolbox: {
        orient: 'vertical',
        top: '25%',
        showTitle: false,
        feature: {
          dataZoom: {
            yAxisIndex: 'none'
          },
          restore: {},
          saveAsImage: {},
          dataView: {}
        },
        tooltip: {
            show: true,
            position: 'left',
            formatter: function (param) {
                return '<div>' + param.title + '</div>';
            },
            backgroundColor: 'darkgray',
            textStyle: {
                fontSize: 12,
                color: 'black'
            },
            extraCssText: 'box-shadow: 0 0 3px rgba(0, 0, 0, 0.3);'
        }
    },
    xAxis: {
        type: 'category',
        boundaryGap: false,
        data: {{chart_id}}_x_labels
    },
    yAxis: {
        name: "NAV",
        nameLocation: 'middle',
        nameGap: '40',
        type: 'value',
        scale: true
    },
    series: {{chart_id}}_get_series(0)
});

window.addEventListener('resize',function(){
    {{chart_id}}.resize();
})

{{chart_id}}.on('restore', function (event) {
    {{chart_id}}_clear_performance_statistics();
});

{{chart_id}}.on('dataZoom', function (event) {
    if(event.batch[0].start == 0 && event.batch[0].end == 100) {
        {{chart_id}}.setOption({
            series: ({{chart_id}}_get_series(0))
        });
        {{chart_id}}_clear_performance_statistics();
    } else {
        {{chart_id}}.setOption({
            series: ({{chart_id}}_get_series(event.batch[0].startValue))
        });
        {{chart_id}}_clear_performance_statistics();
        start_date = {{chart_id}}_x_labels[event.batch[0].startValue]
        end_date = {{chart_id}}_x_labels[event.batch[0].endValue]
        {{chart_id}}_dynamic_content.appendChild(document.createTextNode("Start Date: " + start_date + "; End Date: " + end_date));
        ({{chart_id}}_update_performance_statistics(event.batch[0].startValue, event.batch[0].endValue));
    }
});