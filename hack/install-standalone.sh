
# 1. install istio
kubectl apply -f ./pipeline-standalone/istio/namespace.yaml;
kubectl apply -f ./pipeline-standalone/istio/istio.yaml;
sleep 10;

# 2. install knative
kubectl apply -f ./pipeline-standalone/knative/namespace.yaml;
kubectl apply -f ./pipeline-standalone/knative/serving-crds.yml;
kubectl apply -f ./pipeline-standalone/knative/serving-core.yaml;
kubectl apply -f ./pipeline-standalone/knative/net-istio.yaml;
sleep 10;

# 3. install pipeline standalone
kubectl apply -f ./pipeline-standalone/pipeline/crds.yaml;
kubectl apply -f ./pipeline-standalone/pipeline/namespace.yaml;
kubectl apply -f ./pipeline-standalone/pipeline/deploys.yaml;
sleep 10;

# 4. install paddle operator
kubectl apply -f ./paddle-operator/crds.yaml;
kubectl apply -f ./paddle-operator/deploys.yaml;
sleep 10;

# 5. install jupyter hub
helm repo add jupyterhub https://jupyterhub.github.io/helm-chart/;
helm repo update
helm upgrade --cleanup-on-fail \
  --install jupyterhub/jupyterhub \
  --namespace jhub \
  --create-namespace \
  --version=1.3.0 \
  --values ./pipeline-standalone/jupyterhub/config.yaml
