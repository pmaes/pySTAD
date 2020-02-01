import panel as pn
import vega
pn.extension('vega')

def create_vega_nodes(graph):
    output = []
    # The following will automatically pick up the lens...
    feature_labels = graph.vs()[0].attributes().keys()
    for v in graph.vs():
        node = {"name": v.index}
        for f in feature_labels:
            node[f] = v.attributes()[f]
        output.append(node)
    return output

def create_vega_links(graph):
    output = []
    for e in graph.es():
        output.append({"source": e.source, "target": e.target, "value": e.attributes()['distance']})
    return output

def create_gephi_files(graph, filename):
    with open(filename + '_nodes.csv', 'w') as f:
        if 'lens' in graph.vs()[0].attributes():
            f.write("id\tlens\n")
            counter = 0
            for v in graph.vs():
                f.write(str(counter) + "\t" + str(v.attributes()['lens']) + "\n")
                counter += 1
        else:
            f.write("id\n")
            counter = 0
            for v in graph.vs():
                f.write(str(counter) + "\n")
                counter += 1

    with open(filename + '_edges.csv', 'w') as f:
        f.write("source\ttarget\tvalue\n")
        for e in graph.es():
            f.write(str(e.source) + "\t" + str(e.target) + "\t" + str(e.attributes()['distance']) + "\n")

def draw_stad(graph):
    strength_picker = pn.widgets.IntSlider(name='Attraction strength', start=-10, end=1, step=1, value=-3)
    distance_picker = pn.widgets.IntSlider(name='Distance between nodes', start=1, end=30, step=1, value=15)
    radius_picker = pn.widgets.IntSlider(name='Node radius', start=1, end=5, step=1, value=5)
    theta_picker = pn.widgets.FloatSlider(name='Theta', start=0.1, end=1.5, step=0.1, value=0.9)
    distance_max_picker = pn.widgets.IntSlider(name='Max distance cap', start=0, end=300, step=20, value=100)
    strength_picker.width = 100
    distance_picker.width = 100
    radius_picker.width = 100
    theta_picker.width = 100
    distance_max_picker.width = 100

    @pn.depends(strength_picker.param.value, distance_picker.param.value, radius_picker.param.value, theta_picker.param.value, distance_max_picker.param.value)
    def plot(strength, distance, radius, theta, distance_max):
        nodes = create_vega_nodes(graph)
        links = create_vega_links(graph)
        nodes_string = str(nodes).replace("'", '"')
        links_string = str(links).replace("'", '"')

        tooltip_signal = {}
        feature_labels = nodes[0].keys()
        for f in feature_labels:
            tooltip_signal[f] = "datum['" + f + "']"

        enter = {}
        if 'lens' in nodes[0]:
            enter = {
                "fill": {"field": "lens", "scale": "colour"},
                "tooltip": {"signal": str(tooltip_signal).replace('"','')} # Replace is necessary because no quotes allowed around "datum" (check vega editor)
            }
        else:
            enter = {
                "fill": {"value": "lightgrey"},
                "tooltip": {"signal": str(tooltip_signal).replace('"','')}
            }

        return pn.pane.Vega({
          "$schema": "https://vega.github.io/schema/vega/v5.json",
          "width": 1000,
          "height": 1000,
          "padding": 0,
          "autosize": "none",

          "signals": [
            { "name": "cx", "update": "width / 2" },
            { "name": "cy", "update": "height / 2" },
            {
              "description": "State variable for active node dragged status.",
              "name": "dragged", "value": 0,
              "on": [
                {
                  "events": "symbol:mouseout[!event.buttons], window:mouseup",
                  "update": "0"
                },
                {
                  "events": "symbol:mouseover",
                  "update": "dragged || 1"
                },
                {
                  "events": "[symbol:mousedown, window:mouseup] > window:mousemove!",
                  "update": "2", "force": True
                }
              ]
            },
            {
              "description": "Graph node most recently interacted with.",
              "name": "dragged_node", "value": None,
              "on": [
                {
                  "events": "symbol:mouseover",
                  "update": "dragged === 1 ? item() : dragged_node"
                }
              ]
            },
            {
              "description": "Flag to restart Force simulation upon data changes.",
              "name": "restart", "value": False,
              "on": [
                {"events": {"signal": "dragged"}, "update": "dragged > 1"}
              ]
            }
          ],
          "scales": [
            {
              "name": "colour",
              "type": "linear",
              "domain": {"data": "node-data", "field": "lens"},
              "range": {"scheme": "rainbow"}
            }
          ],

          "data": [
            {
              "name": "node-data",
              "values": nodes
            },
            {
              "name": "link-data",
              "values": links
            }
          ],

          "marks": [
            {
              "name": "nodes",
              "type": "symbol",
              "zindex": 1,

              "from": {"data": "node-data"},
              "on": [
                {
                  "trigger": "dragged",
                  "modify": "dragged_node",
                  "values": "dragged === 1 ? {fx:dragged_node.x, fy:dragged_node.y} : {fx:x(), fy:y()}"
                },
                {
                  "trigger": "!dragged",
                  "modify": "dragged_node", "values": "{fx: null, fy: null}"
                }
              ],

              "encode": {
                "enter": enter,
                "update": {
                  "size": {"value": 50},
                  "cursor": {"value": "pointer"}
                }
              },

              "transform": [
                {
                  "type": "force",
                  "iterations": 300,
                  "velocityDecay": 0.5,
                  "restart": {"signal": "restart"},
                  "static": False,
                  "forces": [
                    {"force": "center", "x": {"signal": "cx"}, "y": {"signal": "cy"}},
                    {"force": "collide", "radius": radius},
                    {"force": "nbody", "strength": strength, "theta": theta, "distanceMax": distance_max},
                    {"force": "link", "links": "link-data", "distance": distance}
                  ]
                }
              ]
            },
            {
              "type": "path",
              "from": {"data": "link-data"},
              "interactive": False,
              "encode": {
                "update": {
                  "stroke": {"value": "lightgrey"}
                }
              },
              "transform": [
                {
                  "type": "linkpath", "shape": "line",
                  "sourceX": "datum.source.x", "sourceY": "datum.source.y",
                  "targetX": "datum.target.x", "targetY": "datum.target.y"
                }
              ]
            }
          ]
        })

    return pn.Row(pn.Column(strength_picker, distance_picker, radius_picker, theta_picker, distance_max_picker), plot).show()
