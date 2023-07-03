var {{chart_id}} = echarts.init(document.getElementById("{{chart_id}}"));

{{chart_id}}.setOption({
    tooltip: {
      trigger: 'axis'
    },
    legend: {
        data: [
            {% for data in datasets %}
            {{data["label"]}},
            {% endfor %}
        ]
    },
    xAxis: {
        type: 'category',
        boundaryGap: false,
        data: {{labels}}
    },
    yAxis: {
        name: "NAV",
        nameLocation: 'middle',
        nameGap: '40',
        type: 'value',
        scale: true
    },
    series: [
        {% for data in datasets %}
        {
            type: 'line',
            name: {{data["label"]}},
            data: {{data["data"]}}
        },
        {% endfor %}
    ]
  });

window.addEventListener('resize',function(){
  {{chart_id}}.resize();
})