var {{chart_id}} = echarts.init(document.getElementById("{{chart_id}}"));
var {{chart_id}}_dynamic_content = document.getElementById("{{chart_id}}_dynamic_content");

{{chart_id}}_type = "{{ nav_type }}"

{{chart_id}}_x_labels = {{x_data}}

{{chart_id}}_returns = {{returns}};

function {{chart_id}}_get_series(start_index) {
    let series = [];
    let nav;
    if({{chart_id}}_type == "compound") {
        nav = get_compount_nav_from_returns({{chart_id}}_returns, start_index);
    } else if ({{chart_id}}_type == "sum") {
        nav = get_sum_nav_from_returns({{chart_id}}_returns, start_index);
    } else {
        throw 'Invalid NAV type ' + {{chart_id}}_type;
    }
    let mdd = get_max_draw_down_from_nav(nav)
    return [
        {
            yAxisIndex: 0,
            name: "NAV",
            type: 'line',
            data: nav
        },
        {
            yAxisIndex: 1,
            name: "MDD",
            type: 'bar',
            data: mdd,
            itemStyle: {
                color: '#ff5c5c',
			  opacity: 0.1
		  }
        }
    ];
};

function {{chart_id}}_clear_performance_statistics() {
    while ({{chart_id}}_dynamic_content.firstChild) {
        {{chart_id}}_dynamic_content.firstChild.remove();
    }
};

function {{chart_id}}_update_performance_statistics(start_index, end_index) {
    performances = [["nav", ...get_performance_statistics_from_returns({{chart_id}}_returns, start_index, end_index)]];
    {{chart_id}}_dynamic_content.appendChild(create_performance_table(performances));
};

{{chart_id}}.setOption({
    tooltip: {
        trigger: 'axis',
        axisPointer: {
            type: 'cross',
            animation: false,
            label: {
                backgroundColor: '#505765'
            }
        }
    },
    toolbox: {
        showTitle: false,
        top: '-2%',
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
    grid: {
        top: '10%',
        bottom: 35,
	},
    legend: {
        data: ['NAV', 'MDD'],
    },
    xAxis: [
        {
            type: 'category',
            boundaryGap: false,
            data: {{chart_id}}_x_labels
        }
    ],
    yAxis: [
        {
            name: 'NAV',
            nameLocation: 'middle',
            nameGap: '45',
            type: 'value',
            scale: true
        },
        {
            name: 'MDD',
            nameLocation: 'middle',
            nameGap: '50',
            type: 'value',
            scale: true,
            axisLabel: {
                  formatter: '{value} %'
            },
            splitLine: {
                  show: false
            }
        }
    ],
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